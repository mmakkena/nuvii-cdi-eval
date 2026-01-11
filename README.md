# Nuvii CDI Agent Evaluation Framework

A comprehensive evaluation framework for the Nuvii Clinical Documentation Improvement (CDI) Agent V2 pipelines. Supports ICD-10 coding, HCC risk adjustment, documentation gap detection, CDI query generation, and E/M level assessment.

## Overview

This framework provides:

- **Offline Evaluation** - Batch runs on labeled datasets with CDI-specific KPIs
- **CI Gating** - Regression testing for prompt/model/retriever changes via Promptfoo
- **Trace-first Debugging** - Auditability via Phoenix tracing with OpenTelemetry
- **RAG Guardrails** - Retrieval and context quality metrics via RAGAS
- **PHI-safe Operation** - Built-in redaction controls and safe defaults
- **Rich Reporting** - HTML, JSON, CSV, and Markdown report generation

## Features

### Evaluators

| Evaluator | Description | Key Metrics |
|-----------|-------------|-------------|
| **ICD** | ICD-10 code suggestion accuracy | Top-N accuracy, F1, hierarchy score |
| **HCC** | HCC detection and RAF scoring | Precision, recall, RAF accuracy |
| **Gap** | Documentation gap detection | Precision, recall, priority accuracy |
| **Query** | CDI query quality assessment | Non-leading score, ACDIS compliance |
| **E/M** | E/M level determination | Exact match, within-1-level, MDM accuracy |

### Integrations

- **Phoenix** - OpenTelemetry-based tracing for debugging and analysis
- **RAGAS** - RAG pipeline quality metrics (context precision, faithfulness)
- **Promptfoo** - CI/CD regression testing with blocking regression detection

## Installation

### Prerequisites

- Python 3.11+
- Poetry (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/mmakkena/nuvii-cdi-eval.git
cd nuvii-cdi-eval

# Install with Poetry (recommended)
poetry install

# Or with pip
pip install -e .

# Copy environment file and configure
cp .env.example .env
# Edit .env with your API credentials
```

## Quick Start

### 1. Configure Environment

Create a `.env` file:

```bash
# API Configuration
NUVII_API_BASE_URL=http://localhost:8000
NUVII_API_KEY=your-api-key
NUVII_API_TIMEOUT=60

# Phoenix Tracing (optional)
PHOENIX_ENABLED=false
PHOENIX_ENDPOINT=http://localhost:6006

# Evaluation Settings
EVAL_PHI_SAFE_MODE=true
```

### 2. Run Evaluations

```bash
# Run evaluation on a dataset
nuvii-eval run eval datasets/examples/icd_coding_tests.json --task icd

# Run all task types
nuvii-eval run eval datasets/examples/ --task all --output results.json

# Dry run to validate dataset
nuvii-eval run eval datasets/examples/icd_coding_tests.json --dry-run
```

### 3. Generate Reports

```bash
# Generate HTML report
nuvii-eval report generate results.json --output report.html --format html

# Generate PR comment for CI
nuvii-eval report pr results.json --output pr_comment.md

# View summary in terminal
nuvii-eval report summary results.json --by-task
```

### 4. Compare Runs for Regressions

```bash
# Compare two evaluation runs
nuvii-eval compare runs baseline.json current.json --threshold 0.05

# Analyze trends over time
nuvii-eval compare trend results/ --last 10
```

## CLI Reference

### Run Commands

```bash
nuvii-eval run eval <dataset> [OPTIONS]
  --task         Task type: icd, hcc, gap, query, em, all (default: all)
  --output       Output file path for results
  --format       Output format: json, csv, html, markdown (default: json)
  --concurrency  Max concurrent API requests (default: 5)
  --timeout      Request timeout in seconds (default: 60)
  --fail-fast    Stop on first failure
  --verbose      Enable verbose output
  --dry-run      Validate without running

nuvii-eval run batch <config.yaml> [OPTIONS]
  --output-dir   Output directory for results (default: ./results)
  --parallel     Run task types in parallel

nuvii-eval run promptfoo <config.yaml> [OPTIONS]
  --output       Output file for results
  --no-cache     Disable result caching
```

### Report Commands

```bash
nuvii-eval report generate <results> [OPTIONS]
  --output       Output file path (default: ./report.html)
  --format       Report format: html, markdown, json, csv, pdf
  --title        Report title
  --details      Include detailed test results (default: true)
  --charts       Include charts (HTML only, default: true)

nuvii-eval report pr <results> [OPTIONS]
  --baseline     Baseline results for comparison
  --output       Output file (default: stdout)
  --max-failures Maximum failed tests to include (default: 10)

nuvii-eval report summary <results> [OPTIONS]
  --by-task      Group results by task type
  --by-specialty Group results by medical specialty
```

### Dataset Commands

```bash
nuvii-eval dataset validate <path> [OPTIONS]
  --strict       Enable strict validation
  --fix          Attempt to fix common issues

nuvii-eval dataset inspect <path> [OPTIONS]
  --samples      Number of sample test cases to show (default: 3)

nuvii-eval dataset convert <input> <output> [OPTIONS]
  --to           Output format: json, yaml

nuvii-eval dataset split <input> [OPTIONS]
  --output       Output directory (default: ./split)
  --train        Training set ratio (default: 0.8)
  --seed         Random seed (default: 42)
```

### Compare Commands

```bash
nuvii-eval compare runs <baseline> <current> [OPTIONS]
  --output       Output file for comparison report
  --threshold    Score drop threshold for regression (default: 0.1)
  --fail-on-regression  Exit with error on blocking regressions

nuvii-eval compare trend <results_dir> [OPTIONS]
  --output       Output file for trend report
  --last         Number of recent runs to analyze (default: 10)

nuvii-eval compare ci [baseline_ref] [OPTIONS]
  --results      Path to current results (default: results/latest.json)
```

## Dataset Format

### ICD Coding Test Case

```json
{
  "id": "icd-001",
  "clinical_note": "65-year-old male presents with...",
  "expected_icd_codes": ["E11.9", "I50.22"],
  "expected_primary_code": "E11.9",
  "specialty": "cardiology",
  "complexity": "moderate",
  "tags": ["diabetes", "heart_failure"]
}
```

### HCC Risk Test Case

```json
{
  "id": "hcc-001",
  "clinical_note": "Patient with chronic conditions...",
  "expected_hcc_codes": ["HCC18", "HCC85"],
  "expected_raf_range": [0.8, 1.2],
  "icd_codes": ["E11.65", "I50.22"],
  "specialty": "endocrinology",
  "complexity": "high"
}
```

### Gap Detection Test Case

```json
{
  "id": "gap-001",
  "clinical_note": "Documentation with missing elements...",
  "expected_gaps": ["Severity not documented", "Laterality missing"],
  "gap_count_range": [2, 4],
  "has_critical_gap": true,
  "specialty": "cardiology",
  "complexity": "moderate"
}
```

### Query Generation Test Case

```json
{
  "id": "query-001",
  "clinical_note": "Clinical note requiring clarification...",
  "query_context": "Reason for query",
  "expected_query_type": "clarification",
  "expected_query_topics": ["specificity", "acuity"],
  "specialty": "pulmonology",
  "complexity": "low"
}
```

### E/M Level Test Case

```json
{
  "id": "em-001",
  "clinical_note": "SOAP note with full documentation...",
  "expected_em_level": "99214",
  "expected_mdm_level": "moderate",
  "mdm_components": {
    "problems": "moderate",
    "data": "moderate",
    "risk": "low"
  },
  "specialty": "primary_care",
  "complexity": "moderate"
}
```

## Example Datasets

Example datasets are provided in `datasets/examples/`:

- `icd_coding_tests.json` - ICD-10 code suggestion test cases
- `hcc_risk_tests.json` - HCC risk adjustment test cases
- `gap_detection_tests.json` - Documentation gap detection test cases
- `query_tests.json` - CDI query generation test cases
- `em_level_tests.json` - E/M level assessment test cases

## KPI Targets

| Metric | Target | Description |
|--------|--------|-------------|
| ICD Top-1 Accuracy | >= 85% | Primary code exact match |
| ICD Top-3 Accuracy | >= 95% | Primary code in top 3 |
| HCC Recall | >= 90% | Captured expected HCCs |
| Gap Detection F1 | >= 80% | Balance of precision/recall |
| Query Quality Score | >= 4.0/5.0 | Rubric composite score |
| E/M Within-1 Level | >= 98% | Within 1 level of expected |

## CI/CD Integration

### GitHub Actions

The framework includes pre-configured workflows in `.github/workflows/`:

- **ci.yml** - Runs on every push/PR: linting, testing, type checking, security scan
- **eval.yml** - Scheduled/manual evaluation runs with regression detection

### Example PR Comment

When run in CI, the framework generates PR comments:

```markdown
## CDI Evaluation Results

| Metric | Value |
|--------|-------|
| Total Tests | 50 |
| Passed | 47 |
| Failed | 3 |
| Pass Rate | 94.0% |

### Regressions Detected: 1
- [MEDIUM] ICD-001: Score dropped from 0.95 to 0.82

### New Failures: 2
- GAP-015: Gap detection failed
- QUERY-008: Query relevance below threshold
```

## Project Structure

```
nuvii-cdi-eval/
├── src/nuvii_eval/
│   ├── cli/                 # Command-line interface
│   │   ├── main.py         # Main CLI app
│   │   └── commands/       # Subcommand modules
│   ├── config.py           # Configuration management
│   ├── client.py           # Nuvii API client
│   ├── datasets/           # Dataset loading and schemas
│   │   ├── schemas.py      # Pydantic models
│   │   ├── loader.py       # Dataset loaders
│   │   └── validator.py    # Validation utilities
│   ├── evaluators/         # Task-specific evaluators
│   │   ├── icd.py          # ICD coding evaluator
│   │   ├── hcc.py          # HCC risk evaluator
│   │   ├── gap.py          # Gap detection evaluator
│   │   ├── query.py        # Query generation evaluator
│   │   └── em.py           # E/M level evaluator
│   ├── instrumentation/    # Observability
│   │   ├── phoenix_tracer.py  # Phoenix/OTel tracing
│   │   └── phi_redactor.py    # PHI redaction
│   ├── promptfoo/          # Promptfoo integration
│   │   ├── converter.py    # Test case converter
│   │   ├── assertions.py   # Custom assertions
│   │   └── regression.py   # Regression detection
│   ├── reporters/          # Report generation
│   │   ├── html_reporter.py
│   │   ├── json_reporter.py
│   │   └── markdown_reporter.py
│   ├── runner/             # Evaluation execution
│   │   ├── batch.py        # Batch runner
│   │   └── async_runner.py # Async parallel runner
│   └── ragas/              # RAGAS integration
│       └── evaluator.py    # RAGAS metrics
├── datasets/               # Test case datasets
│   └── examples/           # Example test cases
├── .github/workflows/      # CI/CD pipelines
├── tests/                  # Unit tests
└── pyproject.toml          # Project configuration
```

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/nuvii_eval --cov-report=html

# Run specific test file
poetry run pytest tests/test_evaluators.py -v
```

### Code Quality

```bash
# Lint with Ruff
poetry run ruff check src/

# Format with Ruff
poetry run ruff format src/

# Type check with mypy
poetry run mypy src/nuvii_eval
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run manually
poetry run pre-commit run --all-files
```

## PHI Safety

This framework is designed with PHI safety in mind:

- **PHI Safe Mode** (default: enabled) - Redacts sensitive data from logs and traces
- **No PHI in Git** - Real clinical data paths are gitignored
- **Synthetic Data** - Sample datasets use synthetic clinical notes

**Important**: Never commit real Protected Health Information to this repository.

## License

Proprietary - Nuvii Health

## Support

For issues and questions, contact the Nuvii engineering team.
