from typing import List, Union, Callable, Dict

# ---------- 1) Summarization ----------

def rouge_scores(generated: str, reference: str) -> Dict[str, float]:
    """
    Returns ROUGE-1/2/L F1 scores.
    Requires: pip install rouge-score
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def bertscore(generated: Union[str, List[str]], reference: Union[str, List[str]]) -> Dict[str, float]:
    """
    Returns BERTScore P/R/F1.
    Requires: pip install bert-score
    """
    from bert_score import score

    cands = [generated] if isinstance(generated, str) else generated
    refs = [reference] if isinstance(reference, str) else reference

    P, R, F1 = score(cands, refs, lang="en", verbose=False)
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
    }


def llm_as_judge(
    generated: str,
    reference: str,
    judge: Callable[[str, str], float],
) -> float:
    """
    Wrapper for an external judge function/model.
    `judge` should return a numeric score (e.g., 0-10 or 0-1).
    """
    return float(judge(generated, reference))


# ---------- 2) Recommendation / Risk Prediction ----------

def precision_score(y_true: List[int], y_pred: List[int]) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall_score(y_true: List[int], y_pred: List[int]) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0