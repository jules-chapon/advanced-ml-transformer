"""Evaluation functions"""

from rouge_score import rouge_scorer
from typing import List, Tuple


def calculate_metrics(
    predicted_sentences: List[str], reference_sentences: List[str]
) -> Tuple[float, float]:
    """
    Calculate the metrics for a list of predicted and reference sentences.
    The metrics are ROUGE-1 and ROUGE-L.

    Args:
        predicted_sentences (list[str]): Predictions of the model.
        reference_sentences (list[str]): References sentences to compare with.

    Returns:
        tuple[float, float]: Metrics (ROUGE-1, ROUGE-L).
    """
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = [
        rouge.score(" ".join(ref[0]), " ".join(pred))
        for ref, pred in zip(reference_sentences, predicted_sentences)
    ]
    avg_rouge1 = sum(score["rouge1"].fmeasure for score in rouge_scores) / len(
        rouge_scores
    )
    avg_rougeL = sum(score["rougeL"].fmeasure for score in rouge_scores) / len(
        rouge_scores
    )
    return avg_rouge1, avg_rougeL
