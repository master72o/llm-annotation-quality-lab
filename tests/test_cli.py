"""
Integration test for quality lab CLI.
"""

import os
import pytest
from quality_lab.cli import load_annotation_dataset
from quality_lab.analyzer import GuidelineCalibrationAnalyzer
from quality_lab.report_generator import ReportGenerator


def test_cli_load_dataset(tmp_path):
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text(
        '{"id": "t1", "prompt": "Hi", "response": "Hello", "ground_truth_label": "no_error", "annotations": [{"annotator_id": "a1", "label": "no_error"}]}\n',
        encoding="utf-8"
    )

    items = load_annotation_dataset(str(jsonl))
    assert len(items) == 1
    assert items[0].id == "t1"


def test_full_calibration_comparison_pipeline(tmp_path):
    b_file = tmp_path / "before.jsonl"
    a_file = tmp_path / "after.jsonl"

    b_file.write_text('{"id": "t1", "prompt": "Hi", "response": "Hello", "ground_truth_label": "no_error", "annotations": [{"annotator_id": "a1", "label": "no_error"}, {"annotator_id": "a2", "label": "factual_error"}]}\n', encoding="utf-8")
    a_file.write_text('{"id": "t1", "prompt": "Hi", "response": "Hello", "ground_truth_label": "no_error", "annotations": [{"annotator_id": "a1", "label": "no_error"}, {"annotator_id": "a2", "label": "no_error"}]}\n', encoding="utf-8")

    before_items = load_annotation_dataset(str(b_file))
    after_items = load_annotation_dataset(str(a_file))

    comparison = GuidelineCalibrationAnalyzer.compare_guideline_impact(before_items, after_items)

    out_dir = tmp_path / "results"
    report_path = tmp_path / "report.md"

    ReportGenerator.export_comparison(comparison, str(out_dir))
    ReportGenerator.generate_markdown_report(comparison, str(report_path))

    assert os.path.exists(out_dir / "annotation_quality_comparison.json")
    assert os.path.exists(report_path)
