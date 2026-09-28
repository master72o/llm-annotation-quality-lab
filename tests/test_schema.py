"""
Unit tests for annotation quality lab data schemas.
"""

import pytest
from quality_lab.schema import FactualErrorLabel, SingleAnnotation, AnnotationTaskItem


def test_factual_error_label_enum():
    assert FactualErrorLabel.NO_ERROR.value == "no_error"
    assert FactualErrorLabel.FACTUAL_ERROR.value == "factual_error"
    assert FactualErrorLabel.UNCERTAIN.value == "uncertain"


def test_single_annotation_serialization():
    ann = SingleAnnotation(
        annotator_id="ann_1",
        label=FactualErrorLabel.FACTUAL_ERROR,
        error_category="wrong_number",
        rationale="Incorrect math calculation",
        is_simulated=False
    )
    d = ann.to_dict()
    assert d["annotator_id"] == "ann_1"
    assert d["label"] == "factual_error"
    assert d["error_category"] == "wrong_number"


def test_task_item_deserialization():
    raw = {
        "id": "t1",
        "prompt": "What is 2+2?",
        "response": "5",
        "ground_truth_label": "factual_error",
        "domain": "Math",
        "annotations": [
            {
                "annotator_id": "ann_1",
                "label": "factual_error",
                "error_category": "math_error",
                "is_simulated": False
            }
        ]
    }
    item = AnnotationTaskItem.from_dict(raw)
    assert item.id == "t1"
    assert item.ground_truth_label == FactualErrorLabel.FACTUAL_ERROR
    assert len(item.annotations) == 1
    assert item.annotations[0].label == FactualErrorLabel.FACTUAL_ERROR
