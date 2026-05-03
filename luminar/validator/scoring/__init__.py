"""
Scoring module with benchmark-specific implementations.

Provides a factory function to get the appropriate scorer for a given benchmark type.
"""

from __future__ import annotations

from dataclasses import dataclass

from luminar.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class ScoreResult:
    final_score: float  # micro F1 (0.0-1.0)
    f1: float  # same as final_score (kept for backward compatibility)
    precision: float
    recall: float
    n_total: int  # total GT boxes across all groups
    n_predicted: int  # total predicted boxes (includes extras)
    n_matched: int  # true-positive boxes (IoU >= 0.5)
    box_f1: float = 0.0
    row_f1: float = 0.0
    item_f1: float = 0.0
    exact_count_ratio: float = 0.0
    n_expected_rows: int = 0
    n_output_rows: int = 0
    n_missing_rows: int = 0
    n_unexpected_rows: int = 0
    n_duplicate_rows: int = 0
    n_extra_rows: int = 0
    n_count_mismatch_rows: int = 0
    n_declared_mismatch_rows: int = 0


def get_scorer(benchmark_type: str):
    """
    Factory function to get the appropriate scorer for a benchmark type.
    
    Args:
        benchmark_type: Type of benchmark ('traffic', 'license-plate', etc.)
    
    Returns:
        Scorer function that takes (output_csv_bytes, ground_truth_bytes) -> ScoreResult
    """
    if benchmark_type == "license-plate":
        from luminar.validator.scoring.license_plate import score_license_plate_output
        return score_license_plate_output
    elif benchmark_type == "traffic":
        from luminar.validator.scoring.traffic import score_output
        return score_output
    else:
        log.warning(
            "Unknown benchmark type '%s', defaulting to traffic scorer",
            benchmark_type,
        )
        from luminar.validator.scoring.traffic import score_output
        return score_output


__all__ = [
    "ScoreResult",
    "get_scorer",
]
