# Datasets

This directory contains test case datasets for evaluating the Nuvii CDI Agent.

## Directory Structure

```
datasets/
├── golden/           # Gold standard labeled test cases
│   ├── icd_test_cases.jsonl
│   ├── hcc_test_cases.jsonl
│   ├── gap_test_cases.jsonl
│   ├── query_test_cases.jsonl
│   ├── em_test_cases.jsonl
│   └── cpt_test_cases.jsonl
├── regression/       # Fast CI regression subset
│   └── fast_suite.jsonl
├── synthetic/        # Auto-generated test cases
└── samples/          # Sample data for development
```

## Dataset Format

All datasets use JSONL (JSON Lines) format with one test case per line.

### ICD Test Case Example

```json
{
  "id": "icd_001",
  "clinical_note": "72-year-old female with type 2 diabetes mellitus...",
  "specialty": "endocrinology",
  "complexity": "moderate",
  "expected_icd_codes": ["E11.9"],
  "acceptable_icd_codes": ["E11.65"],
  "tags": ["diabetes", "chronic"]
}
```

### HCC Test Case Example

```json
{
  "id": "hcc_001",
  "clinical_note": "65-year-old male with CHF, LVEF 35%...",
  "specialty": "cardiology",
  "complexity": "high",
  "expected_hccs": ["HCC85"],
  "expected_raf_range": [0.3, 0.5],
  "patient_age": 65,
  "patient_gender": "M"
}
```

### Gap Test Case Example

```json
{
  "id": "gap_001",
  "clinical_note": "Patient with diabetes, recent lab shows...",
  "specialty": "endocrinology",
  "complexity": "moderate",
  "expected_gaps": [
    {
      "gap_type": "missing_specificity",
      "condition": "diabetes complications",
      "min_priority": 2
    }
  ]
}
```

## PHI Safety

**IMPORTANT**: This directory should NEVER contain real Protected Health Information (PHI).

- Use synthetic or de-identified data only
- Real clinical data paths are gitignored
- Review all commits for accidental PHI inclusion

## Creating New Test Cases

1. Use the appropriate schema from `src/nuvii_eval/datasets/schemas.py`
2. Validate with the dataset loader before committing
3. Include appropriate tags for filtering
4. Document the source (synthetic, curated, etc.)

```python
from nuvii_eval.datasets import DatasetLoader

loader = DatasetLoader("./datasets")
stats = loader.validate_dataset("golden/icd_test_cases.jsonl", "icd")
print(stats)
```

## Minimum Dataset Requirements

| Type | Minimum Cases | Description |
|------|---------------|-------------|
| ICD  | 200 | ICD-10 coding accuracy |
| HCC  | 100 | HCC/RAF scoring |
| Gap  | 150 | Documentation gap detection |
| Query | 100 | CDI query quality |
| E/M  | 150 | E/M level accuracy |
| CPT  | 100 | Procedure coding |
| Regression | 50 | Fast CI subset |
