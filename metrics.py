"""Extra evaluation metrics: BERTScore, BLEURT, NLI entailment, entity preservation.

Models are lazily loaded and cached on first use so eval_model() can call these
functions once per run without re-downloading/re-loading weights.
"""
import torch
import torch.nn.functional as F
from functools import lru_cache

BERTSCORE_MODEL_TYPE = "roberta-large"
BLEURT_MODEL_NAME = "Elron/bleurt-base-128"
NLI_MODEL_NAME = "facebook/bart-large-mnli"

# NLTK resources needed for ne_chunk-based NER (no extra pip dependency beyond nltk,
# already used for corpus_bleu elsewhere in this repo)
_NLTK_RESOURCES = [
    ("punkt", "tokenizers/punkt"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    ("maxent_ne_chunker", "chunkers/maxent_ne_chunker"),
    ("words", "corpora/words"),
]


@lru_cache(maxsize=1)
def _get_nli():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def _get_bleurt():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(BLEURT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BLEURT_MODEL_NAME)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def _get_ner():
    import nltk
    for name, path in _NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)
    return nltk


def _extract_entities(nltk_mod, text):
    tree = nltk_mod.ne_chunk(nltk_mod.pos_tag(nltk_mod.word_tokenize(text)))
    return {
        " ".join(word for word, _ in subtree.leaves()).strip().lower()
        for subtree in tree
        if hasattr(subtree, "label")
    }


def _non_empty_pairs(candidates, references):
    return [(c, r) for c, r in zip(candidates, references) if c.strip() and r.strip()]


def compute_bertscore(candidates, references, device="cpu"):
    """Mean precision/recall/F1 BERTScore over the batch. Empty strings are skipped."""
    from bert_score import score as bertscore_score

    pairs = _non_empty_pairs(candidates, references)
    if not pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    cands, refs = zip(*pairs)
    P, R, F1 = bertscore_score(
        list(cands), list(refs), lang="en", model_type=BERTSCORE_MODEL_TYPE,
        device=device, verbose=False,
    )
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}


def compute_bleurt(candidates, references, device="cpu", batch_size=16):
    """Mean BLEURT score (torch-only HF port, no TensorFlow needed)."""
    tokenizer, model = _get_bleurt()
    model.to(device)

    scores = []
    for i in range(0, len(candidates), batch_size):
        batch_cands = candidates[i:i + batch_size]
        batch_refs = references[i:i + batch_size]
        inputs = tokenizer(batch_refs, batch_cands, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            batch_scores = model(**inputs).logits.flatten()
        scores.extend(batch_scores.cpu().tolist())
    return sum(scores) / len(scores) if scores else 0.0


def compute_entailment(candidates, references, device="cpu", batch_size=16):
    """NLI entailment of candidate given reference as premise (does candidate follow from reference?).

    Returns mean entailment probability and the fraction of pairs whose argmax label is entailment.
    """
    tokenizer, model = _get_nli()
    model.to(device)

    label2id = {k.lower(): v for k, v in model.config.label2id.items()}
    entail_idx = label2id["entailment"]

    entail_probs = []
    entail_hits = 0
    for i in range(0, len(candidates), batch_size):
        batch_cands = candidates[i:i + batch_size]
        batch_refs = references[i:i + batch_size]
        inputs = tokenizer(batch_refs, batch_cands, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
        entail_probs.extend(probs[:, entail_idx].cpu().tolist())
        entail_hits += (probs.argmax(dim=-1) == entail_idx).sum().item()

    n = len(candidates)
    return {
        "entailment_prob": sum(entail_probs) / n if n else 0.0,
        "entailment_accuracy": entail_hits / n if n else 0.0,
    }


def compute_entity_preservation(candidates, references):
    """Fraction of named entities in the reference that also appear (surface form) in the candidate,
    averaged over samples that contain at least one reference entity.
    """
    nltk_mod = _get_ner()

    per_sample_scores = []
    for cand, ref in zip(candidates, references):
        ref_entities = _extract_entities(nltk_mod, ref)
        if not ref_entities:
            continue
        cand_lower = cand.lower()
        preserved = sum(1 for ent in ref_entities if ent in cand_lower)
        per_sample_scores.append(preserved / len(ref_entities))

    return sum(per_sample_scores) / len(per_sample_scores) if per_sample_scores else 0.0
