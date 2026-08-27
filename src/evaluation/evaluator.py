"""Offline recall evaluation for the legal QA engine."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Protocol, Sequence


class PipelineEngine(Protocol):
    def run_pipeline(self, query: str) -> dict[str, Any]: ...


class Evaluator:
    def __init__(self, engine: PipelineEngine):
        self.engine = engine

    def calculate_recall(self, predicted_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        if not ground_truth_ids:
            return 0.0
        hits = sum(1 for predicted_id in predicted_ids if predicted_id in ground_truth_ids)
        return hits / len(ground_truth_ids)

    def run_evaluation(self, csv_path: str | Path) -> dict[str, int | float]:
        """Evaluate every CSV row and return a consumable count/mean-recall summary."""
        total_recall = 0.0
        count = 0
        with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    raise ValueError("evaluation rows must contain a question and ground-truth IDs")
                query, truth_string = row[0], row[1]
                truths = [truth.strip() for truth in truth_string.split(",") if truth.strip()]
                response = self.engine.run_pipeline(query)
                predicted = response["relevant_articles"]
                total_recall += self.calculate_recall(predicted, truths)
                count += 1

        return {"count": count, "mean_recall": total_recall / count if count else 0.0}
