"""Evaluation functions"""

from rouge_score import rouge_scorer


def calculate_metrics(
    predicted_sentences: list[str], reference_sentences: list[str]
) -> tuple[float, float, float, float]:
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
