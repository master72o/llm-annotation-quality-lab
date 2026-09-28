"""
Before vs After Guidelines Calibration & Disagreement Analyzer.
"""

from typing import List, Dict, Any
from quality_lab.schema import AnnotationTaskItem
from quality_lab.metrics import AgreementMetrics


class GuidelineCalibrationAnalyzer:
    """Analyzes impact of annotation guidelines on agreement metrics."""

    @staticmethod
    def compare_guideline_impact(
        before_items: List[AnnotationTaskItem],
        after_items: List[AnnotationTaskItem]
    ) -> Dict[str, Any]:
        before_metrics = AgreementMetrics.calculate_dataset_metrics(before_items)
        after_metrics = AgreementMetrics.calculate_dataset_metrics(after_items)

        delta_fleiss_k = after_metrics["fleiss_kappa"] - before_metrics["fleiss_kappa"]
        delta_kripp_a = after_metrics["krippendorff_alpha"] - before_metrics["krippendorff_alpha"]
        delta_raw_agree = after_metrics["raw_agreement_rate"] - before_metrics["raw_agreement_rate"]
        delta_acc = after_metrics["accuracy_against_ground_truth"] - before_metrics["accuracy_against_ground_truth"]

        return {
            "before_guidelines": before_metrics,
            "after_guidelines": after_metrics,
            "improvements": {
                "delta_fleiss_kappa": round(delta_fleiss_k, 4),
                "delta_krippendorff_alpha": round(delta_kripp_a, 4),
                "delta_raw_agreement_rate": round(delta_raw_agree, 4),
                "delta_accuracy_gt": round(delta_acc, 4),
            }
        }
