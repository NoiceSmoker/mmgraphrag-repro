"""Text-LLM access for the KG agents and the GRASP filter.

``CachedLLM`` batches all requests of a pipeline stage into a single vLLM
call and memoises every answer in SQLite (keyed by model + prompt +
decoding parameters + sample seed), which makes every stage resumable and
makes re-running a stage free.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable, Protocol

from mmrag.common.cache import KVCache
from mmrag.common.io import stable_hash

_THINK = re.compile(r"<think>.*?</think>", flags=re.S)


@dataclass(frozen=True)
class ChatRequest:
    system: str
    user: str
    max_tokens: int = 256
    temperature: float = 0.0
    seed: int = 0  # distinguishes re-samples of the same prompt (inspector re-runs)


class Backend(Protocol):
    name: str

    def generate(self, reqs: list[ChatRequest]) -> list[str]: ...


class VLLMBackend:
    """vLLM offline engine. Qwen3 models are run in non-thinking mode."""

    def __init__(self, model: str, max_model_len: int = 4096, gpu_memory_utilization: float = 0.85,
                 quantization: str | None = None, enable_thinking: bool = False, max_num_seqs: int = 128):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.name = model
        self.enable_thinking = enable_thinking
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        kw = dict(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                  trust_remote_code=True, max_num_seqs=max_num_seqs)
        if quantization:
            kw["quantization"] = quantization
        self.llm = LLM(**kw)

    def _prompt(self, r: ChatRequest) -> str:
        msgs = ([{"role": "system", "content": r.system}] if r.system else []) + [{"role": "user", "content": r.user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=self.enable_thinking)
        except TypeError:  # templates without the enable_thinking switch
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, reqs: list[ChatRequest]) -> list[str]:
        from vllm import SamplingParams

        params = [SamplingParams(temperature=r.temperature, top_p=0.8 if r.temperature > 0 else 1.0,
                                 max_tokens=r.max_tokens, seed=r.seed if r.temperature > 0 else None)
                  for r in reqs]
        outs = self.llm.generate([self._prompt(r) for r in reqs], sampling_params=params, use_tqdm=True)
        return [_THINK.sub("", o.outputs[0].text).strip() for o in outs]


class FunctionBackend:
    """Deterministic backend for tests: ``fn(system, user) -> str``."""

    def __init__(self, fn: Callable[[str, str], str], name: str = "fake"):
        self.fn = fn
        self.name = name
        self.calls = 0

    def generate(self, reqs: list[ChatRequest]) -> list[str]:
        self.calls += len(reqs)
        return [self.fn(r.system, r.user) for r in reqs]


class CachedLLM:
    def __init__(self, backend: Backend, cache: KVCache | None):
        self.backend = backend
        self.cache = cache
        self.n_calls = 0      # requests actually sent to the model
        self.n_cached = 0

    def _key(self, r: ChatRequest) -> str:
        return "llm:" + stable_hash({"m": self.backend.name, **asdict(r)}, n=24)

    def complete(self, reqs: list[ChatRequest]) -> list[str]:
        if not reqs:
            return []
        keys = [self._key(r) for r in reqs]
        hit = self.cache.get_many(keys) if self.cache is not None else {}
        todo_idx, seen = [], {}
        for i, k in enumerate(keys):
            if k not in hit and k not in seen:
                seen[k] = i
                todo_idx.append(i)
        if todo_idx:
            outs = self.backend.generate([reqs[i] for i in todo_idx])
            new = {keys[i]: o for i, o in zip(todo_idx, outs)}
            if self.cache is not None:
                self.cache.set_many(new)
            hit.update(new)
        self.n_calls += len(todo_idx)
        self.n_cached += len(reqs) - len(todo_idx)
        return [hit[k] for k in keys]


def make_llm(model: str, cache: KVCache | None, **kw) -> CachedLLM:
    return CachedLLM(VLLMBackend(model, **kw), cache)
