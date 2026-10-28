"""Audio-side models: CLAP embeddings (retrieval) and text-to-audio grounding.

* ``ClapEncoder``   LAION-CLAP through transformers; audio and text are
                    projected into a shared space (paper ref. [48]).
* ``TAGScorer``     weakly-supervised text-to-audio grounding model of
                    Xu et al. (paper ref. [51]); returns the frame-wise
                    probability that a phrase is audible and we take the
                    maximum over time as the presence score Phi_a.
* ``ClapWindowScorer`` fallback grounding if the TAG checkpoint cannot be
                    loaded: max cosine similarity between the phrase and
                    2-second CLAP windows.
"""
from __future__ import annotations

import inspect

import numpy as np

from mmrag.m3kg.audio import load_audio


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_tensor(out):
    """transformers 4.x returns tensors from get_*_features; some newer
    versions return model-output objects."""
    import torch

    if isinstance(out, torch.Tensor):
        return out
    for attr in ("text_embeds", "audio_embeds", "pooler_output", "last_hidden_state"):
        if getattr(out, attr, None) is not None:
            return getattr(out, attr)
    raise TypeError(f"cannot extract embeddings from {type(out)}")


class ClapEncoder:
    SR = 48_000

    def __init__(self, model_id: str = "laion/larger_clap_general", device: str | None = None):
        from transformers import ClapModel, ClapProcessor

        self.device = device or _device()
        self.model = ClapModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = ClapProcessor.from_pretrained(model_id)
        params = inspect.signature(self.proc.__call__).parameters
        self._audio_kw = "audio" if "audio" in params else "audios"

    def embed_waveforms(self, waves: list[np.ndarray], batch: int = 32) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(waves), batch):
            inp = self.proc(**{self._audio_kw: waves[i : i + batch]}, sampling_rate=self.SR, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with torch.no_grad():
                e = _as_tensor(self.model.get_audio_features(**inp))
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 512), np.float32)

    def embed_audio_files(self, paths: list[str], batch: int = 32, max_seconds: float = 10.0) -> np.ndarray:
        out = []
        for i in range(0, len(paths), batch):
            waves = [load_audio(p, self.SR, max_seconds) for p in paths[i : i + batch]]
            out.append(self.embed_waveforms(waves, batch))
            if (i // batch) % 20 == 0:
                print(f"[clap] audio {i + len(waves)}/{len(paths)}")
        return np.concatenate(out, axis=0)

    def embed_text(self, texts: list[str], batch: int = 128) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(texts), batch):
            inp = self.proc(text=texts[i : i + batch], padding=True, truncation=True, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with torch.no_grad():
                e = _as_tensor(self.model.get_text_features(**inp))
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 512), np.float32)


class TAGScorer:
    name = "tag"

    def __init__(self, model_id: str = "wsntxxn/cnn8rnn-laionclap-audiocapsv2-grounding", device: str | None = None):
        from transformers import AutoModel

        self.device = device or _device()
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
        self.sr = int(self.model.config.sample_rate)

    def score(self, audio_path: str, phrases: list[str], batch: int = 64) -> np.ndarray:
        """Presence score in [0, 1] for each phrase (max over time)."""
        import torch

        wav = torch.from_numpy(load_audio(audio_path, self.sr)).to(self.device)
        scores = []
        for i in range(0, len(phrases), batch):
            chunk = phrases[i : i + batch]
            audio = wav.unsqueeze(0).expand(len(chunk), -1).contiguous()
            with torch.no_grad():
                out = self.model(audio=audio, audio_len=[wav.shape[0]] * len(chunk), text=chunk)
            if isinstance(out, dict):
                out = next(v for v in out.values() if isinstance(v, torch.Tensor))
            out = out.float()
            if out.dim() == 1:
                out = out.unsqueeze(-1)
            scores.append(out.reshape(len(chunk), -1).max(dim=-1).values.cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0, np.float32)


class ClapWindowScorer:
    name = "clap_window"

    def __init__(self, clap: ClapEncoder, win_s: float = 2.0, hop_s: float = 1.0):
        self.clap, self.win, self.hop = clap, win_s, hop_s

    def score(self, audio_path: str, phrases: list[str], batch: int = 64) -> np.ndarray:
        sr = ClapEncoder.SR
        wav = load_audio(audio_path, sr, 10.0)
        w, h = int(self.win * sr), int(self.hop * sr)
        windows = [wav[s : s + w] for s in range(0, max(1, len(wav) - w + 1), h)] or [wav]
        a = self.clap.embed_waveforms(windows)
        t = self.clap.embed_text(phrases)
        return (t @ a.T).max(axis=1)


def make_grounder(cfg: dict, clap: ClapEncoder):
    g = cfg["grasp"]
    if g.get("grounding", "tag") == "tag":
        try:
            return TAGScorer(g["tag_model"])
        except Exception as exc:  # remote code / version mismatch
            print(f"[grasp] TAG model unavailable ({exc}); falling back to CLAP windows")
    return ClapWindowScorer(clap)
