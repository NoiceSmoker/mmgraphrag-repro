"""Prompts of the M3KG-RAG agents.

The paper lists its prompts in the supplementary material (Tables 5-14).
No official code is available, so the prompts below re-express every rule of
those tables in our own words; deviations are marked with "NOTE".
"""

# --------------------------------------------------------------------------
# Step 1 - context-enriched triplet extraction
# --------------------------------------------------------------------------
REWRITER_SYSTEM = """You improve the caption of a short audio clip using the title and description of the video it was cut from.
Rules:
- The original caption has priority. If the title/description do not clearly describe the same scene, object or action, return the original caption unchanged.
- When they clearly match, you may only (a) replace a generic noun with a more specific term (breed, species, instrument, vehicle model, place, role) or (b) add one short factual attribute.
- Never invent events, numbers, counts or speculative facts; never add advertisements, links or hashtags.
- Keep the meaning, style and approximate length (within +/-20%) of the original caption, and merge the information naturally.
Return only the final English caption."""

REWRITER_USER = "Title: {title}\nDescription: {description}\nOriginal caption: {caption}\nCaption:"

EXTRACTOR_SYSTEM = """You extract structured knowledge from a caption of an audio or video clip.
Return every meaningful (head, relation, tail) triple stated or directly implied by the caption.
- Use short surface forms that occur in (or are directly implied by) the caption.
- Do not add entities or facts that the caption does not support.
- Write exactly one triple per line in the form (head, relation, tail), in English, with no other text."""

EXTRACTOR_USER = "Caption: {caption}\nTriples:"

# --------------------------------------------------------------------------
# Step 2 - knowledge grounding
# --------------------------------------------------------------------------
NORMALIZER_SYSTEM = """Map a phrase to one concept name that could be looked up in an encyclopedia (a Wikipedia-style title).
- Output a single English noun phrase in plain text: no quotes, no explanation.
- If the phrase is not English, translate it to an English title (transliterate proper names).
- Otherwise use only words from the phrase, in their original order; you may drop words but not add or translate them.
- Drop articles, possessives, quantities, years and wrappers such as "how to" or "the sound of"; prefer the inner object noun phrase.
- Use the singular form and keep the casing of proper names."""

NORMALIZER_USER = "Phrase: {mention}\nConcept:"

CALLBACK_SYSTEM = """You write short, neutral encyclopedia definitions.
Write one or two sentences that define the given concept in general terms.
The caption is given only to tell which sense of the concept is meant; do not describe the caption's scene.
Return only the definition sentences."""

CALLBACK_USER = "Concept: {concept}\nCaption (for disambiguation only): {caption}\nDefinition:"

# --------------------------------------------------------------------------
# Step 3 - context-aware description refinement
# --------------------------------------------------------------------------
SELECTOR_SYSTEM = """You pick the encyclopedic description that matches the sense in which a concept is used in a clip caption.
- Use only the caption to decide the sense; do not bring in outside facts.
- If several candidates fit, prefer the most specific one that is not speculative; if none fits clearly, prefer the safest generic one.
Answer with the number of the chosen candidate only."""
# NOTE: the paper asks the selector to copy the chosen candidate verbatim;
# asking for its number selects the same text but cannot corrupt it.

SELECTOR_USER = "Concept: {concept}\nCaption: {caption}\nCandidates:\n{candidates}\nChosen candidate number:"

REFINER_SYSTEM = """You adapt a description written for a canonical concept so that it fits the original wording of an entity.
- Change as little as possible (determiners, possessives, number, surface phrasing) so that the description reads naturally for the original wording.
- Do not add, remove or invent facts and do not change the meaning.
- Return one or two plain sentences, without lists, quotes or reasoning."""

REFINER_USER = ("Original wording: {mention}\nCanonical concept: {concept}\n"
                "Description of the canonical concept: {description}\nAdapted description:")

INSPECTOR_SYSTEM = """You judge whether an encyclopedic description has the intended sense of an entity.
Rate from 0 (different or irrelevant sense) to 10 (exactly the intended sense).
The caption shows the context in which the entity occurs.
Answer with one integer between 0 and 10 and nothing else."""
# NOTE: Table 11 shows only CONCEPT and DESCRIPTION; Sec. A.1.1 says the
# score is conditioned on the context-enriched caption, so we pass it too.

INSPECTOR_USER = "Entity: {mention}\nCaption: {caption}\nDescription: {description}\nScore:"

# --------------------------------------------------------------------------
# GRASP - selective pruning with an LLM
# --------------------------------------------------------------------------
GRASP_FILTER_SYSTEM = """You remove knowledge triples that are not needed to answer a question about an audio clip.
- Keep every triple that could help to answer the question.
- Remove only triples that are clearly irrelevant, that contradict the question, or that duplicate another triple.
- When unsure, keep the triple.
Return a JSON list with the indices of the kept triples, in their original order, and nothing else."""

GRASP_FILTER_USER = "Question: {question}\nTriples:\n{triples}\nKept indices:"

# --------------------------------------------------------------------------
# Graph-augmented generation (paper Eq. 6 / Table 13)
# --------------------------------------------------------------------------
RAG_INSTRUCTION = """You answer questions about the audio you hear.
The audio itself is your primary evidence. Below you also get knowledge triples retrieved from a knowledge graph; treat them only as optional hints.
1. Check whether a triple (its entities, attributes, actions, time or place) is actually present in the audio.
2. If it is, you may use the whole triple and the entity descriptions to give a more specific answer. Never contradict what you hear; ignore conflicting triples.
3. If no triple clearly matches, answer from the audio alone.

Question: {question}

Retrieved triples:
{triples}"""

CAPTION_INSTRUCTION = """You answer questions about the audio you hear.
The audio itself is your primary evidence. Below you also get captions of other recordings that sound similar; treat them only as optional hints.
1. Check whether what a caption describes is actually present in the audio.
2. If it is, you may use it to give a more specific answer. Never contradict what you hear; ignore conflicting captions.
3. If no caption clearly matches, answer from the audio alone.

Question: {question}

Captions of similar recordings:
{captions}"""

TRIPLE_LINE = "[{i}] head={h} | relation={r} | tail={t} || head_description={hd} | tail_description={td}"
