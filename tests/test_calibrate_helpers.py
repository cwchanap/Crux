"""Regression tests for calibrate_egmd_mapping helper functions."""

import pytest

from src.benchmark.models import ScoreSummary


def test_score_summary_dict_coalesces_none_to_zero():
    """score_summary_dict must convert None precision/recall/f1 to 0.0.

    When a bin has zero predictions for a class, ScoreSummary.f1 is None.
    The calibration pipeline calls float() and <= on these dict values, so
    None must be coalesced to avoid TypeError at runtime.
    """
    from scripts.calibrate_egmd_mapping import score_summary_dict

    # All-zero counts -> precision=None, recall=None, f1=None
    summary = ScoreSummary(true_positives=0, false_positives=0, false_negatives=0)
    result = score_summary_dict(summary)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_score_summary_dict_preserves_real_values():
    from scripts.calibrate_egmd_mapping import score_summary_dict

    summary = ScoreSummary(true_positives=8, false_positives=2, false_negatives=2)
    result = score_summary_dict(summary)
    assert result["precision"] == 0.8
    assert result["recall"] == 0.8
    assert result["f1"] == pytest.approx(0.8)


def test_strategy_sort_key_handles_coalesced_zeros():
    """strategy_sort_key must not raise on metrics derived from empty bins."""
    from scripts.calibrate_egmd_mapping import score_summary_dict, strategy_sort_key

    summary = ScoreSummary(true_positives=0, false_positives=0, false_negatives=0)
    metrics = score_summary_dict(summary)
    # Should not raise TypeError
    key = strategy_sort_key(metrics)
    assert key == (0.0, 0, 0.0, 0.0)


def test_strategy_sort_key_orders_real_metrics_above_zeros():
    from scripts.calibrate_egmd_mapping import score_summary_dict, strategy_sort_key

    zero_metrics = score_summary_dict(
        ScoreSummary(true_positives=0, false_positives=0, false_negatives=0)
    )
    real_metrics = score_summary_dict(
        ScoreSummary(true_positives=5, false_positives=1, false_negatives=1)
    )
    assert strategy_sort_key(real_metrics) > strategy_sort_key(zero_metrics)


def test_f1_comparison_does_not_raise_on_coalesced_zero():
    """build_candidate_mappings compares metrics['f1'] <= 0.0 — must not crash."""
    from scripts.calibrate_egmd_mapping import score_summary_dict

    summary = ScoreSummary(true_positives=0, false_positives=0, false_negatives=0)
    metrics = score_summary_dict(summary)
    assert metrics["f1"] <= 0.0
