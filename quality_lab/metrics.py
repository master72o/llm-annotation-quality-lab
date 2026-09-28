"""
Statistical Agreement Metrics Module: Krippendorff's Alpha, Fleiss' Kappa, Cohen's Kappa.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from quality_lab.schema import AnnotationTaskItem, FactualErrorLabel, SingleAnnotation


class AgreementMetrics:
    """Calculates inter-annotator agreement statistics."""

    @staticmethod
    def compute_krippendorff_alpha_nominal(matrix: np.ndarray) -> float:
        """Compute Krippendorff's Alpha for nominal ratings.

        Args:
            matrix: N x K matrix where N is items, K is categories (counts of annotators per category).
        """
        N, K = matrix.shape
        units_n = np.sum(matrix, axis=1)  # Number of ratings per unit
        
        # Total number of pairs per unit
        pairs_per_unit = units_n * (units_n - 1)
        valid_units = units_n > 1
        if not np.any(valid_units):
            return 1.0

        total_pairs = np.sum(pairs_per_unit[valid_units])
        if total_pairs == 0:
            return 1.0

        # Observed disagreement (Do)
        # For nominal data, coincidence matrix
        n_total = np.sum(matrix)
        coincidence = np.zeros((K, K))
        
        for i in range(N):
            ni = units_n[i]
            if ni > 1:
                row = matrix[i, :]
                for c1 in range(K):
                    for c2 in range(K):
                        if c1 == c2:
                            coincidence[c1, c2] += row[c1] * (row[c1] - 1)
                        else:
                            coincidence[c1, c2] += row[c1] * row[c2]

        # Observed disagreement ratio Do
        # Off-diagonal sum normalized
        off_diag_obs = np.sum(coincidence) - np.trace(coincidence)
        D_o = off_diag_obs / total_pairs

        # Expected disagreement ratio De
        marginal_k = np.sum(coincidence, axis=1)
        total_coincidence_pairs = np.sum(marginal_k)
        if total_coincidence_pairs == 0:
            return 1.0

        D_e = 0.0
        for c1 in range(K):
            for c2 in range(K):
                if c1 != c2:
                    D_e += (marginal_k[c1] * marginal_k[c2]) / (total_coincidence_pairs * (total_coincidence_pairs - 1))

        if D_e == 0:
            return 1.0

        alpha = 1.0 - (D_o / D_e)
        return float(alpha)

    @staticmethod
    def compute_fleiss_kappa(matrix: np.ndarray) -> float:
        N, K = matrix.shape
        n = np.sum(matrix[0, :])
        if n <= 1:
            return 1.0

        p = np.sum(matrix, axis=0) / (N * n)
        P_i = (np.sum(matrix**2, axis=1) - n) / (n * (n - 1))
        P_bar = np.mean(P_i)
        P_e_bar = np.sum(p**2)

        if P_e_bar == 1.0:
            return 1.0

        kappa = (P_bar - P_e_bar) / (1.0 - P_e_bar)
        return float(kappa)

    @staticmethod
    def compute_cohens_kappa(r1: List[str], r2: List[str]) -> float:
        if len(r1) != len(r2) or len(r1) == 0:
            return 0.0

        categories = sorted(list(set(r1).union(set(r2))))
        cat_map = {c: i for i, c in enumerate(categories)}
        k = len(categories)

        cm = np.zeros((k, k), dtype=int)
        for a, b in zip(r1, r2):
            cm[cat_map[a], cat_map[b]] += 1

        total = len(r1)
        po = np.trace(cm) / total
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total**2)

        if pe == 1.0:
            return 1.0

        kappa = (po - pe) / (1.0 - pe)
        return float(kappa)

    @classmethod
    def calculate_dataset_metrics(cls, items: List[AnnotationTaskItem]) -> Dict[str, Any]:
        if not items:
            return {"total_items": 0, "fleiss_kappa": 0.0, "krippendorff_alpha": 0.0}

        categories = ["no_error", "factual_error", "uncertain"]
        cat_map = {"no_error": 0, "factual_error": 1, "uncertain": 2}

        rating_matrix = []
        raw_agree_count = 0
        total_multi = 0
        correct_against_gt = 0
        total_annotations_count = 0

        disagreements_by_domain: Dict[str, int] = {}
        disagreements_by_category: Dict[str, int] = {}

        for item in items:
            counts = [0, 0, 0]
            labels = []

            for ann in item.annotations:
                lbl_val = ann.label.value if isinstance(ann.label, FactualErrorLabel) else ann.label
                labels.append(lbl_val)
                counts[cat_map[lbl_val]] += 1
                total_annotations_count += 1

                gt_val = item.ground_truth_label.value if isinstance(item.ground_truth_label, FactualErrorLabel) else item.ground_truth_label
                if lbl_val == gt_val:
                    correct_against_gt += 1

            if len(item.annotations) > 1:
                total_multi += 1
                unique_labels = set(labels)
                if len(unique_labels) == 1:
                    raw_agree_count += 1
                else:
                    disagreements_by_domain[item.domain] = disagreements_by_domain.get(item.domain, 0) + 1
                    for ann in item.annotations:
                        if ann.error_category:
                            disagreements_by_category[ann.error_category] = disagreements_by_category.get(ann.error_category, 0) + 1

                rating_matrix.append(counts)

        mat = np.array(rating_matrix) if rating_matrix else np.zeros((0, 3))
        fleiss_k = cls.compute_fleiss_kappa(mat) if mat.shape[0] > 0 else 0.0
        kripp_a = cls.compute_krippendorff_alpha_nominal(mat) if mat.shape[0] > 0 else 0.0

        raw_agree_rate = raw_agree_count / total_multi if total_multi > 0 else 1.0
        disagree_rate = 1.0 - raw_agree_rate
        accuracy_gt = correct_against_gt / total_annotations_count if total_annotations_count > 0 else 0.0

        return {
            "total_items": len(items),
            "total_annotations": total_annotations_count,
            "fleiss_kappa": round(fleiss_k, 4),
            "krippendorff_alpha": round(kripp_a, 4),
            "raw_agreement_rate": round(raw_agree_rate, 4),
            "disagreement_rate": round(disagree_rate, 4),
            "accuracy_against_ground_truth": round(accuracy_gt, 4),
            "disagreements_by_domain": disagreements_by_domain,
            "disagreements_by_category": disagreements_by_category,
        }
