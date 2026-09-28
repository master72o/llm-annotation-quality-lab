"""
Data Schemas for LLM Annotation Quality Research Lab.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


class FactualErrorLabel(str, Enum):
    NO_ERROR = "no_error"
    FACTUAL_ERROR = "factual_error"
    UNCERTAIN = "uncertain"


@dataclass
class SingleAnnotation:
    annotator_id: str
    label: FactualErrorLabel
    error_category: Optional[str] = None
    rationale: str = ""
    is_simulated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotator_id": self.annotator_id,
            "label": self.label.value if isinstance(self.label, FactualErrorLabel) else self.label,
            "error_category": self.error_category,
            "rationale": self.rationale,
            "is_simulated": self.is_simulated,
        }


@dataclass
class AnnotationTaskItem:
    id: str
    prompt: str
    response: str
    ground_truth_label: FactualErrorLabel
    annotations: List[SingleAnnotation]
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationTaskItem":
        annotations = []
        for ann in data.get("annotations", []):
            lbl = FactualErrorLabel(ann["label"])
            annotations.append(
                SingleAnnotation(
                    annotator_id=ann["annotator_id"],
                    label=lbl,
                    error_category=ann.get("error_category"),
                    rationale=ann.get("rationale", ""),
                    is_simulated=ann.get("is_simulated", False),
                )
            )

        gt = FactualErrorLabel(data["ground_truth_label"])

        return cls(
            id=data["id"],
            prompt=data["prompt"],
            response=data["response"],
            ground_truth_label=gt,
            annotations=annotations,
            domain=data.get("domain", "general"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "response": self.response,
            "ground_truth_label": self.ground_truth_label.value if isinstance(self.ground_truth_label, FactualErrorLabel) else self.ground_truth_label,
            "domain": self.domain,
            "annotations": [ann.to_dict() for ann in self.annotations],
            "metadata": self.metadata,
        }
