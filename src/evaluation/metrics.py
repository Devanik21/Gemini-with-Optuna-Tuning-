"""
Metrics for evaluating Gemini model predictions.
"""

def rouge_l(prediction: str, reference: str) -> float:
    """Calculates a simulated ROUGE-L score."""
    if not prediction or not reference:
        return 0.0
    pred_words = prediction.lower().split()
    ref_words = reference.lower().split()
    common = set(pred_words).intersection(set(ref_words))
    return len(common) / max(len(ref_words), 1)

def exact_match(prediction: str, reference: str) -> float:
    """Calculates exact match score."""
    return 1.0 if prediction.strip() == reference.strip() else 0.0
