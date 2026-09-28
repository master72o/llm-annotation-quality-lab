"""
Visualization Generator for Annotation Quality Research Reports.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class Visualizer:
    """Generates comparison charts for annotation quality research."""

    @staticmethod
    def generate_all_figures(comparison: Dict[str, Any], output_dir: str = "reports/figures") -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        generated = {}

        # 1. Before vs After Agreement Comparison
        agree_path = os.path.join(output_dir, "before_vs_after_agreement.png")
        Visualizer._plot_before_vs_after_agreement(
            comparison["before_guidelines"],
            comparison["after_guidelines"],
            agree_path
        )
        generated["before_vs_after_agreement"] = agree_path

        # 2. Disagreement by Category
        cat_path = os.path.join(output_dir, "disagreement_by_category.png")
        Visualizer._plot_disagreement_by_category(
            comparison["before_guidelines"].get("disagreements_by_category", {}),
            comparison["after_guidelines"].get("disagreements_by_category", {}),
            cat_path
        )
        generated["disagreement_by_category"] = cat_path

        # 3. Ground Truth Accuracy Improvement
        acc_path = os.path.join(output_dir, "ground_truth_accuracy.png")
        Visualizer._plot_accuracy_comparison(
            comparison["before_guidelines"].get("accuracy_against_ground_truth", 0.0),
            comparison["after_guidelines"].get("accuracy_against_ground_truth", 0.0),
            acc_path
        )
        generated["ground_truth_accuracy"] = acc_path

        return generated

    @staticmethod
    def _plot_before_vs_after_agreement(before: Dict[str, Any], after: Dict[str, Any], output_path: str):
        fig, ax = plt.subplots(figsize=(9, 5))
        metrics_labels = ["Fleiss' Kappa (κ)", "Krippendorff's Alpha (α)", "Raw Agreement Rate"]
        before_vals = [before.get("fleiss_kappa", 0.0), before.get("krippendorff_alpha", 0.0), before.get("raw_agreement_rate", 0.0)]
        after_vals = [after.get("fleiss_kappa", 0.0), after.get("krippendorff_alpha", 0.0), after.get("raw_agreement_rate", 0.0)]

        x = np.arange(len(metrics_labels))
        width = 0.35

        rects1 = ax.bar(x - width/2, before_vals, width, label="Before Guidelines", color="#d9534f")
        rects2 = ax.bar(x + width/2, after_vals, width, label="After Guidelines & Calibration", color="#5cb85c")

        ax.set_ylabel("Score / Agreement Rate", fontsize=11, fontweight="bold")
        ax.set_title("Annotator Agreement Metrics: Before vs After Guidelines", fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels)
        ax.set_ylim(-0.2, 1.1)
        ax.axhline(y=0.70, color="#2b5c8f", linestyle="--", label="Target Quality Threshold (0.70)")
        ax.legend(loc="upper left")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for rect in rects1:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _plot_disagreement_by_category(before_cat: Dict[str, int], after_cat: Dict[str, int], output_path: str):
        fig, ax = plt.subplots(figsize=(10, 5))
        all_cats = sorted(list(set(before_cat.keys()).union(set(after_cat.keys()))))

        if not all_cats:
            ax.text(0.5, 0.5, "No Category Disagreements Recorded", ha="center", va="center", fontsize=12)
        else:
            b_cnts = [before_cat.get(c, 0) for c in all_cats]
            a_cnts = [after_cat.get(c, 0) for c in all_cats]

            x = np.arange(len(all_cats))
            width = 0.35

            rects1 = ax.bar(x - width/2, b_cnts, width, label="Before Guidelines", color="#d9534f")
            rects2 = ax.bar(x + width/2, a_cnts, width, label="After Guidelines", color="#5bc0de")

            ax.set_ylabel("Disagreement Frequency", fontsize=11, fontweight="bold")
            ax.set_title("Category-Specific Disagreement Breakdown", fontsize=13, fontweight="bold", pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(all_cats, rotation=30, ha="right")
            ax.legend()
            ax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _plot_accuracy_comparison(before_acc: float, after_acc: float, output_path: str):
        fig, ax = plt.subplots(figsize=(7, 5))
        stages = ["Before Guidelines", "After Guidelines"]
        accs = [before_acc * 100, after_acc * 100]
        colors = ["#d9534f", "#5cb85c"]

        bars = ax.bar(stages, accs, color=colors, edgecolor="#333333", width=0.4)
        ax.set_ylabel("Accuracy against Ground Truth (%)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_title("Annotator Accuracy Improvement against Ground Truth", fontsize=13, fontweight="bold", pad=15)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
