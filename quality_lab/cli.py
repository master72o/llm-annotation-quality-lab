"""
Command-Line Interface for LLM Annotation Quality Lab.
"""

import os
import json
import argparse
from typing import List
from quality_lab.schema import AnnotationTaskItem
from quality_lab.analyzer import GuidelineCalibrationAnalyzer
from quality_lab.report_generator import ReportGenerator
from quality_lab.visualizer import Visualizer


def load_annotation_dataset(file_path: str) -> List[AnnotationTaskItem]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    items: List[AnnotationTaskItem] = []
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    items.append(AnnotationTaskItem.from_dict(data))
        else:
            data_list = json.load(f)
            for data in data_list:
                items.append(AnnotationTaskItem.from_dict(data))
    return items


def main():
    parser = argparse.ArgumentParser(description="LLM Annotation Quality Lab CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser("compare", help="Compare Before vs After guidelines datasets")
    compare_parser.add_argument("--before", "-b", required=True, help="Path to Before Guidelines dataset (.jsonl)")
    compare_parser.add_argument("--after", "-a", required=True, help="Path to After Guidelines dataset (.jsonl)")
    compare_parser.add_argument("--output-dir", "-o", default="results", help="Directory to save JSON results")
    compare_parser.add_argument("--report", "-r", default="reports/summary_report.md", help="Path to generate report")
    compare_parser.add_argument("--figures-dir", "-f", default="reports/figures", help="Directory to save figures")

    args = parser.parse_args()

    if args.command == "compare":
        print(f"Loading 'Before Guidelines' dataset: {args.before}")
        before_items = load_annotation_dataset(args.before)
        print(f"Loaded {len(before_items)} items.")

        print(f"Loading 'After Guidelines' dataset: {args.after}")
        after_items = load_annotation_dataset(args.after)
        print(f"Loaded {len(after_items)} items.")

        print("\nAnalyzing inter-annotator agreement improvements (Fleiss' Kappa, Krippendorff's Alpha, Accuracy)...")
        comparison = GuidelineCalibrationAnalyzer.compare_guideline_impact(before_items, after_items)

        imp = comparison["improvements"]
        print(f" Fleiss' Kappa (κ): {comparison['before_guidelines']['fleiss_kappa']:.4f} -> {comparison['after_guidelines']['fleiss_kappa']:.4f} (Δ {imp['delta_fleiss_kappa']:+.4f})")
        print(f" Krippendorff's Alpha (α): {comparison['before_guidelines']['krippendorff_alpha']:.4f} -> {comparison['after_guidelines']['krippendorff_alpha']:.4f} (Δ {imp['delta_krippendorff_alpha']:+.4f})")
        print(f" Accuracy vs Ground Truth: {comparison['before_guidelines']['accuracy_against_ground_truth']*100:.1f}% -> {comparison['after_guidelines']['accuracy_against_ground_truth']*100:.1f}% (Δ {imp['delta_accuracy_gt']*100:+.1f}%)")

        print("\nExporting comparison results...")
        ReportGenerator.export_comparison(comparison, args.output_dir)

        print("\nGenerating visual report figures...")
        Visualizer.generate_all_figures(comparison, args.figures_dir)

        print(f"\nGenerating Markdown research report at: {args.report}")
        ReportGenerator.generate_markdown_report(comparison, args.report)

        print("\nAnnotation Quality Study Completed Successfully!")


if __name__ == "__main__":
    main()
