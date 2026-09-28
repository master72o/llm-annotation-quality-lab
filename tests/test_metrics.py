"""
Unit tests for Krippendorff's Alpha, Fleiss' Kappa, and agreement metrics.
"""

import pytest
import numpy as np
from quality_lab.metrics import AgreementMetrics
from quality_lab.schema import AnnotationTaskItem, FactualErrorLabel, SingleAnnotation


def test_krippendorff_alpha_perfect_agreement():
    # 3 items, 3 categories, 2 annotators per item agree 100%
    matrix = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2]
    ])
    alpha = AgreementMetrics.compute_krippendorff_alpha_nominal(matrix)
    assert alpha == 1.0


def test_fleiss_kappa_perfect_agreement():
    matrix = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2]
    ])
    kappa = AgreementMetrics.compute_fleiss_kappa(matrix)
    assert kappa == 1.0


def test_cohens_kappa_perfect_agreement():
    r1 = ["factual_error", "no_error", "no_error"]
    r2 = ["factual_error", "no_error", "no_error"]
    kappa = AgreementMetrics.compute_cohens_kappa(r1, r2)
    assert kappa == 1.0


def test_dataset_metrics_calculation():
    item = AnnotationTaskItem.from_dict({
        "id": "t1",
        "prompt": "Capital of France?",
        "response": "Paris",
        "ground_truth_label": "no_error",
        "domain": "QA",
        "annotations": [
            {"annotator_id": "ann_1", "label": "no_error"},
            {"annotator_id": "ann_2", "label": "no_error"}
        ]
    })
    metrics = AgreementMetrics.calculate_dataset_metrics([item])
    assert metrics["total_items"] == 1
    assert metrics["fleiss_kappa"] == 1.0
    assert metrics["krippendorff_alpha"] == 1.0
    assert metrics["accuracy_against_ground_truth"] == 1.0
