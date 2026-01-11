# Nuvii CDI Agent Evaluation Framework

A comprehensive evaluation framework for the Nuvii CDI Agent V2 pipelines, supporting ICD, CPT, E/M, HCC, and gap detection evaluation.

## Overview

This framework provides:

- **Offline evaluation** - Batch runs on labeled datasets with CDI-specific KPIs
- **CI gating** - Regression testing for prompt/model/retriever changes via Promptfoo
- **Trace-first debugging** - Auditability via Phoenix tracing
- **RAG guardrails** - Retrieval and context quality metrics via RAGAS
- **PHI-safe operation** - Built-in redaction controls and safe defaults

## Features

### Evaluators

| Evaluator | Description | Key Metrics |
|-----------|-------------|-------------|
| **ICD** | ICD-10 code suggestion accuracy | Top-N accuracy, F1, hierarchy score |
| **HCC** | HCC detection and RAF scoring | Precision, recall, RAF accuracy |
| **Gap** | Documentation gap detection | Precision, recall, priority accuracy |
| **Query** | CDI query quality assessment | Non-leading score, ACDIS compliance |
| **E/M** | E/M level determination | Exact match, within-1-level, MDM accuracy |
| **CPT** | Procedure code suggestion | Precision, modifier accuracy |

### Integrations

- **Phoenix** - OpenTelemetry-based tracing for debugging and analysis
- **RAGAS** - RAG pipeline quality metrics (context precision, faithfulness)
- **Promptfoo** - CI/CD regression testing and red-teaming

## Installation

### Prerequisites

- Python 3.11+
- Poetry (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/nuvii-cdi-eval.git
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

```bash
# Set your Nuvii API credentials
export NUVII_API_URL=http://your-api-url:8000
export NUVII_API_KEY=your_api_key

# Optional: Configure Phoenix tracing
export PHOENIX_ENABLED=true
export PHOENIX_ENDPOINT=http://localhost:6006
```

### 2. Run Evaluation

```bash
# Run ICD evaluation on a dataset
poetry run python scripts/run_eval.py run \
    --dataset datasets/golden/icd_test_cases.jsonl \
    --evaluators icd \
    --output runs/

# Run multiple evaluators
poetry run python scripts/run_eval.py run \
    --dataset datasets/golden/ \
    --evaluators icd,hcc,gap,query
```

### 3. View Results

```bash
# Generate report
poetry run python scripts/run_eval.py report --run-id 20240115_120000

# View in Phoenix UI (if enabled)
open http://localhost:6006
```

## Project Structure

```
nuvii-cdi-eval/
├── src/nuvii_eval/
│   ├── config.py              # Configuration management
│   ├── client.py              # Nuvii API client
│   ├── schemas/               # API response schemas
│   ├── datasets/              # Test case schemas & loaders
│   ├── evaluators/            # Evaluation logic
│   ├── instrumentation/       # Phoenix tracing & PHI redaction
│   ├── runner/                # Batch evaluation runner
│   └── reporters/             # Output formatters
├── promptfoo/                 # CI regression configuration
├── datasets/                  # Test case datasets
│   ├── golden/                # Gold standard labeled data
│   ├── regression/            # Fast CI subset
│   └── synthetic/             # Generated test cases
├── runs/                      # Evaluation outputs (gitignored)
├── scripts/                   # CLI entrypoints
└── tests/                     # Unit tests
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NUVII_API_URL` | Nuvii API base URL | `http://localhost:8000` |
| `NUVII_API_KEY` | API authentication key | (required) |
| `PHOENIX_ENABLED` | Enable Phoenix tracing | `true` |
| `EVAL_PHI_SAFE_MODE` | Enable PHI redaction | `true` |
| `EVAL_CONCURRENCY` | Max concurrent requests | `5` |

See [.env.example](.env.example) for full configuration options.

## Dataset Format

Test cases are stored in JSONL format. Example ICD test case:

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

See [datasets/README.md](datasets/README.md) for detailed schema documentation.

## KPI Targets

| Metric | Target | Description |
|--------|--------|-------------|
| ICD Top-1 Accuracy | ≥ 85% | Primary code exact match |
| ICD Top-3 Accuracy | ≥ 95% | Primary code in top 3 |
| HCC Recall | ≥ 90% | Captured expected HCCs |
| Gap Detection F1 | ≥ 80% | Balance of precision/recall |
| Query Quality Score | ≥ 4.0/5.0 | Rubric composite score |
| E/M Within-1 Level | ≥ 98% | Within 1 level of expected |

## CI/CD Integration

### GitHub Actions

The framework includes pre-configured workflows:

- **ci-regression.yml** - Runs on PRs, fast regression subset
- **ci-full-eval.yml** - Weekly full evaluation suite

### Promptfoo

```bash
# Run Promptfoo regression tests
cd promptfoo
npx promptfoo eval --config promptfooconfig.yaml
```

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/nuvii_eval

# Run specific test file
poetry run pytest tests/test_evaluators.py
```

### Code Quality

```bash
# Run linting
poetry run ruff check src/

# Run type checking
poetry run mypy src/

# Format code
poetry run ruff format src/
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

## Documentation

- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) - Detailed development roadmap
- [Evaluation Framework Plan](docs/EVALUATION_FRAMEWORK_PLAN.md) - Architecture overview
- [Dataset Guide](datasets/README.md) - Test case schema documentation

## License

Proprietary - Nuvii Health

## Support

For issues and questions, contact the Nuvii engineering team.
