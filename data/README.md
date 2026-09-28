# Annotation Quality Lab Data Governance & Experiment Setup

## Dataset Metadata

- **Name**: LLM Factual Error Annotation Benchmark (`before_guidelines.jsonl` vs `after_guidelines.jsonl`)
- **Version**: 1.0.0
- **Format**: JSON Lines (`.jsonl`)
- **Task Definition**: *"Does this model response contain a factual error?"* (`no_error`, `factual_error`, `uncertain`)
- **Sample Count**: 20 prompt-response pairs per dataset
- **Annotators per Item**: 3 independent annotators per prompt item (60 total annotation instances per phase)
- **License**: Creative Commons Attribution 4.0 International (CC-BY-4.0)

## Experimental Setup: Before vs. After Guidelines

- **Phase 1 (`before_guidelines.jsonl`)**: Annotators were given raw prompts and responses without explicit definitions or edge-case handling guidelines. Resulted in high disagreement due to subjective interpretations and unverified claims.
- **Phase 2 (`after_guidelines.jsonl`)**: Annotators underwent calibration training, used reference verification tools, and followed strict guidelines. Resulted in substantial inter-annotator agreement ($\kappa \ge 0.85$, $\alpha \ge 0.85$).

## Data Disclosure

Annotators marked with `"is_simulated": true` represent synthetic baseline annotators constructed to simulate uncalibrated human noise distributions in Controlled Experiment 1.
