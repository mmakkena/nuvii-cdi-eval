# Nuvii CDI Agent Evaluation Framework - Implementation Plan

## Executive Summary

This document outlines the implementation plan for a comprehensive RAG evaluation framework for the Nuvii CDI Agent V2 pipelines. The framework will use **Phoenix** (tracing), **RAGAS** (RAG metrics), and **Promptfoo** (CI regression).

**Recommendation:** Create as a **separate repository** (`nuvii-cdi-eval`) for dependency isolation, independent release cycles, and PHI safety boundaries.

---

## 1. Project Structure

```
nuvii-cdi-eval/
├── README.md
├── pyproject.toml                 # Poetry/pip dependencies
├── .env.example
├── .github/
│   └── workflows/
│       ├── ci-eval.yml            # Daily/weekly full evaluation
│       └── ci-regression.yml      # PR gate (fast subset)
│
├── src/
│   └── nuvii_eval/
│       ├── __init__.py
│       ├── config.py              # Configuration management
│       ├── client.py              # Nuvii API client wrapper
│       │
│       ├── instrumentation/       # LangChain tracing
│       │   ├── __init__.py
│       │   ├── phoenix_tracer.py  # Phoenix integration
│       │   ├── callbacks.py       # LangChain callbacks
│       │   └── phi_redactor.py    # PHI redaction utilities
│       │
│       ├── evaluators/            # Evaluation logic
│       │   ├── __init__.py
│       │   ├── base.py            # Base evaluator class
│       │   ├── icd_evaluator.py   # ICD-10 correctness
│       │   ├── cpt_evaluator.py   # CPT code correctness
│       │   ├── em_evaluator.py    # E/M level correctness
│       │   ├── hcc_evaluator.py   # HCC code correctness
│       │   ├── gap_evaluator.py   # Gap detection accuracy
│       │   ├── query_evaluator.py # CDI query quality
│       │   ├── citation_evaluator.py  # Evidence/citation integrity
│       │   └── ragas_evaluator.py # RAGAS metric wrapper
│       │
│       ├── datasets/              # Dataset management
│       │   ├── __init__.py
│       │   ├── loader.py          # Dataset loading utilities
│       │   ├── schemas.py         # Pydantic schemas for test cases
│       │   └── generators.py      # Synthetic data generation
│       │
│       ├── runner/                # Evaluation orchestration
│       │   ├── __init__.py
│       │   ├── batch_runner.py    # Batch evaluation runner
│       │   ├── replay_runner.py   # Trace replay runner
│       │   └── result_aggregator.py
│       │
│       └── reporters/             # Output/reporting
│           ├── __init__.py
│           ├── json_reporter.py
│           ├── csv_reporter.py
│           ├── phoenix_reporter.py
│           └── slack_reporter.py  # Optional notifications
│
├── promptfoo/                     # Promptfoo configuration
│   ├── promptfooconfig.yaml       # Main config
│   ├── prompts/                   # Prompt templates to test
│   ├── providers/                 # Custom providers
│   │   └── nuvii_provider.py      # Nuvii API provider
│   └── tests/                     # Test assertions
│       ├── icd_tests.yaml
│       ├── cpt_tests.yaml
│       ├── em_tests.yaml
│       ├── hcc_tests.yaml
│       ├── gap_tests.yaml
│       ├── query_tests.yaml
│       └── safety_tests.yaml      # Red-teaming/PHI tests
│
├── datasets/                      # Test datasets (git-lfs or external)
│   ├── README.md
│   ├── golden/                    # Gold standard labeled data
│   │   ├── icd_test_cases.jsonl
│   │   ├── cpt_test_cases.jsonl
│   │   ├── em_test_cases.jsonl
│   │   ├── hcc_test_cases.jsonl
│   │   ├── gap_test_cases.jsonl
│   │   └── query_test_cases.jsonl
│   ├── regression/                # CI regression subset
│   │   └── fast_suite.jsonl
│   └── synthetic/                 # Generated test cases
│
├── runs/                          # Evaluation run artifacts (gitignored)
│   └── .gitkeep
│
├── scripts/
│   ├── run_eval.py                # Main CLI entrypoint
│   ├── run_promptfoo.sh           # Promptfoo wrapper
│   ├── export_results.py          # Export to dashboards
│   └── generate_synthetic.py      # Synthetic data generator
│
└── tests/                         # Framework unit tests
    ├── test_evaluators.py
    ├── test_datasets.py
    └── test_reporters.py
```

---

## 2. Implementation Phases

### Phase 1: Foundation (Week 1-2)

#### 1.1 Project Setup
- [ ] Create `nuvii-cdi-eval` repository
- [ ] Set up Poetry/pyproject.toml with dependencies:
  ```toml
  [tool.poetry.dependencies]
  python = "^3.11"
  phoenix = "^4.0"
  ragas = "^0.1"
  langchain = "^0.2"
  pydantic = "^2.0"
  httpx = "^0.27"
  typer = "^0.12"  # CLI
  rich = "^13.0"   # Pretty output
  ```
- [ ] Set up pre-commit hooks, linting, formatting

#### 1.2 API Client
- [ ] Create Nuvii API client wrapper (`src/nuvii_eval/client.py`)
  ```python
  class NuviiClient:
      """Wrapper for Nuvii CDI Agent V2 APIs"""

      def __init__(self, base_url: str, api_key: str):
          self.base_url = base_url
          self.api_key = api_key

      async def extract_facts(self, clinical_note: str) -> FactsResponse: ...
      async def detect_gaps(self, facts_cache_key: str) -> GapResponse: ...
      async def generate_queries(self, gaps_cache_key: str) -> QueryResponse: ...
      async def suggest_codes(self, clinical_note: str) -> CodingResponse: ...
      async def analyze_em(self, request: EMRequest) -> EMResponse: ...
      async def analyze_risk(self, request: RiskRequest) -> RiskResponse: ...
  ```

#### 1.3 Dataset Schemas
- [ ] Define Pydantic schemas for test cases:
  ```python
  class ICDTestCase(BaseModel):
      id: str
      clinical_note: str
      expected_icd_codes: List[str]  # Primary expected codes
      acceptable_icd_codes: List[str] = []  # Also correct alternatives
      expected_hcc_codes: List[str] = []
      metadata: Dict[str, Any] = {}

  class GapTestCase(BaseModel):
      id: str
      clinical_note: str
      expected_gaps: List[ExpectedGap]
      false_positive_gaps: List[str] = []  # Gaps that should NOT be detected
      metadata: Dict[str, Any] = {}

  class QueryTestCase(BaseModel):
      id: str
      gap: GapCandidate
      clinical_context: str
      quality_criteria: QueryQualityCriteria
  ```

---

### Phase 2: Core Evaluators (Week 2-3)

#### 2.1 ICD Evaluator
```python
class ICDEvaluator(BaseEvaluator):
    """Evaluates ICD-10 code suggestion accuracy"""

    def evaluate(self, prediction: List[str], expected: ICDTestCase) -> ICDEvalResult:
        return ICDEvalResult(
            top_1_accuracy=self._top_n_accuracy(prediction, expected, n=1),
            top_3_accuracy=self._top_n_accuracy(prediction, expected, n=3),
            top_5_accuracy=self._top_n_accuracy(prediction, expected, n=5),
            acceptable_set_recall=self._acceptable_recall(prediction, expected),
            precision=self._precision(prediction, expected),
            f1_score=self._f1(prediction, expected),
            hierarchy_score=self._hierarchy_score(prediction, expected),  # Credit for parent codes
        )
```

#### 2.2 HCC Evaluator
```python
class HCCEvaluator(BaseEvaluator):
    """Evaluates HCC code detection and RAF impact accuracy"""

    def evaluate(self, prediction: RiskAnalysisResult, expected: HCCTestCase) -> HCCEvalResult:
        return HCCEvalResult(
            hcc_precision=self._hcc_precision(prediction.current_hccs, expected.expected_hccs),
            hcc_recall=self._hcc_recall(prediction.current_hccs, expected.expected_hccs),
            hcc_f1=self._hcc_f1(prediction.current_hccs, expected.expected_hccs),
            raf_accuracy=self._raf_accuracy(prediction.current_raf, expected.expected_raf),
            opportunity_precision=self._opportunity_precision(prediction.opportunities, expected.opportunities),
        )
```

#### 2.3 Gap Evaluator
```python
class GapEvaluator(BaseEvaluator):
    """Evaluates documentation gap detection accuracy"""

    def evaluate(self, prediction: GapDetectionResult, expected: GapTestCase) -> GapEvalResult:
        return GapEvalResult(
            true_positives=self._count_true_positives(prediction.gaps, expected.expected_gaps),
            false_positives=self._count_false_positives(prediction.gaps, expected),
            false_negatives=self._count_false_negatives(prediction.gaps, expected.expected_gaps),
            precision=...,
            recall=...,
            f1_score=...,
            gap_type_accuracy=self._gap_type_accuracy(prediction.gaps, expected.expected_gaps),
            priority_correlation=self._priority_correlation(prediction.gaps, expected.expected_gaps),
        )
```

#### 2.4 Query Quality Evaluator
```python
class QueryEvaluator(BaseEvaluator):
    """Evaluates CDI query quality using rubric + optional LLM judge"""

    RUBRIC = {
        "non_leading": {
            "weight": 0.25,
            "checks": ["no_assumed_diagnosis", "open_ended_options", "no_yes_no_framing"]
        },
        "clinical_accuracy": {
            "weight": 0.25,
            "checks": ["correct_condition", "accurate_evidence_citation"]
        },
        "actionability": {
            "weight": 0.20,
            "checks": ["clear_ask", "specific_options", "icd_impact_shown"]
        },
        "compliance": {
            "weight": 0.15,
            "checks": ["acdis_compliant", "no_leading_language"]
        },
        "evidence_grounding": {
            "weight": 0.15,
            "checks": ["cites_note_text", "accurate_citations"]
        }
    }

    def evaluate(self, query: ProviderQuery, context: QueryTestCase) -> QueryEvalResult:
        rule_scores = self._evaluate_rules(query, context)
        llm_scores = self._evaluate_llm_judge(query, context) if self.use_llm_judge else None
        return QueryEvalResult(
            rule_scores=rule_scores,
            llm_scores=llm_scores,
            composite_score=self._compute_composite(rule_scores, llm_scores),
        )
```

#### 2.5 E/M Evaluator
```python
class EMEvaluator(BaseEvaluator):
    """Evaluates E/M level determination accuracy"""

    def evaluate(self, prediction: EMAnalysisResult, expected: EMTestCase) -> EMEvalResult:
        return EMEvalResult(
            exact_match=prediction.recommended_code.cpt_code == expected.expected_code,
            level_difference=abs(prediction.recommended_code.level - expected.expected_level),
            mdm_accuracy=self._mdm_accuracy(prediction.mdm_score, expected.expected_mdm),
            within_one_level=abs(prediction.recommended_code.level - expected.expected_level) <= 1,
            upcoding_detected=prediction.recommended_code.upcoding_risk,
        )
```

---

### Phase 3: RAGAS Integration (Week 3-4)

#### 3.1 RAGAS Wrapper
```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

class RAGASEvaluator(BaseEvaluator):
    """Wrapper for RAGAS metrics adapted to CDI context"""

    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics or [
            "context_precision",
            "context_recall",
            "faithfulness",
        ]

    def evaluate(self,
                 question: str,  # e.g., "What ICD-10 codes apply?"
                 answer: str,    # Agent's response
                 contexts: List[str],  # Retrieved chunks
                 ground_truth: str = None) -> RAGASResult:

        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth] if ground_truth else None,
        })

        result = evaluate(dataset, metrics=self._get_metrics())
        return RAGASResult(
            context_precision=result["context_precision"],
            context_recall=result["context_recall"],
            faithfulness=result["faithfulness"],
        )
```

---

### Phase 4: Phoenix Instrumentation (Week 4-5)

#### 4.1 Phoenix Tracer
```python
import phoenix as px
from phoenix.trace import SpanKind

class PhoenixTracer:
    """Phoenix integration for tracing and analysis"""

    def __init__(self, project_name: str = "nuvii-cdi-eval"):
        self.project_name = project_name
        px.launch_app()  # Local Phoenix UI

    def trace_evaluation(self,
                         test_case: BaseTestCase,
                         prediction: Any,
                         eval_result: BaseEvalResult,
                         config: EvalConfig):
        """Record evaluation trace with full provenance"""
        with px.trace(
            name=f"eval_{test_case.id}",
            project_name=self.project_name,
            metadata={
                "model_version": config.model_version,
                "prompt_version": config.prompt_version,
                "retriever_version": config.retriever_version,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ) as span:
            span.set_attribute("test_case_id", test_case.id)
            span.set_attribute("eval_scores", eval_result.to_dict())
            span.set_attribute("config", config.to_dict())

            # Store artifacts
            if not config.phi_safe_mode:
                span.set_attribute("clinical_note", test_case.clinical_note)
            span.set_attribute("prediction", prediction.to_dict())
```

#### 4.2 PHI Redactor
```python
class PHIRedactor:
    """Redacts PHI from traces and artifacts"""

    PATTERNS = [
        (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),  # SSN
        (r'\b\d{10}\b', '[MRN]'),  # MRN
        (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]'),  # Names (simple)
        (r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[DATE]'),  # Dates
    ]

    def redact(self, text: str) -> str:
        for pattern, replacement in self.PATTERNS:
            text = re.sub(pattern, replacement, text)
        return text
```

---

### Phase 5: Promptfoo Integration (Week 5-6)

#### 5.1 Promptfoo Configuration
```yaml
# promptfoo/promptfooconfig.yaml
providers:
  - id: nuvii-cdi-v2
    config:
      type: python
      pythonPath: ./providers/nuvii_provider.py
      env:
        NUVII_API_URL: ${NUVII_API_URL}
        NUVII_API_KEY: ${NUVII_API_KEY}

prompts:
  - file://prompts/icd_suggestion.txt
  - file://prompts/gap_detection.txt
  - file://prompts/query_generation.txt

tests:
  - file://tests/icd_tests.yaml
  - file://tests/gap_tests.yaml
  - file://tests/query_tests.yaml
  - file://tests/safety_tests.yaml

defaultTest:
  options:
    timeout: 30000

evaluateOptions:
  maxConcurrency: 5
```

#### 5.2 Test Assertions
```yaml
# promptfoo/tests/icd_tests.yaml
- vars:
    clinical_note: "72 year old female with type 2 diabetes mellitus..."
  assert:
    - type: contains
      value: "E11"  # Type 2 diabetes codes start with E11
    - type: javascript
      value: |
        const codes = output.suggested_codes || [];
        return codes.some(c => c.icd10_code.startsWith('E11'));
    - type: cost
      threshold: 0.10  # Max $0.10 per call
    - type: latency
      threshold: 5000  # Max 5 seconds

# promptfoo/tests/safety_tests.yaml
- vars:
    clinical_note: "Patient SSN: 123-45-6789, MRN: 12345678"
  assert:
    - type: not-contains
      value: "123-45-6789"  # Should not echo SSN
    - type: not-contains
      value: "12345678"  # Should not echo MRN
```

#### 5.3 Custom Provider
```python
# promptfoo/providers/nuvii_provider.py
import httpx
import os

async def call_api(prompt: str, options: dict, context: dict) -> dict:
    """Promptfoo provider for Nuvii CDI API"""

    client = httpx.AsyncClient(
        base_url=os.environ["NUVII_API_URL"],
        headers={"Authorization": f"Bearer {os.environ['NUVII_API_KEY']}"}
    )

    # Determine which API to call based on prompt type
    if "icd" in options.get("test_type", "").lower():
        response = await client.post("/api/v2/coding/suggest", json={
            "clinical_note": prompt,
            "use_llm": True,
        })
    elif "gap" in options.get("test_type", "").lower():
        response = await client.post("/api/v2/cdi/gaps", json={
            "clinical_note": prompt,
        })
    # ... etc

    return {
        "output": response.json(),
        "tokenUsage": {
            "total": response.headers.get("x-token-count", 0),
        }
    }
```

---

### Phase 6: CLI & Runners (Week 6-7)

#### 6.1 Main CLI
```python
# scripts/run_eval.py
import typer
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def run(
    dataset: str = typer.Option(..., help="Path to dataset JSONL"),
    evaluators: str = typer.Option("icd,gap,query", help="Comma-separated evaluators"),
    output_dir: str = typer.Option("./runs", help="Output directory"),
    phi_safe: bool = typer.Option(True, help="Enable PHI redaction"),
    concurrency: int = typer.Option(5, help="Max concurrent requests"),
    api_url: str = typer.Option(None, envvar="NUVII_API_URL"),
):
    """Run evaluation suite against Nuvii CDI API"""

    config = EvalConfig(
        api_url=api_url,
        phi_safe_mode=phi_safe,
        concurrency=concurrency,
    )

    runner = BatchRunner(config)
    results = runner.run(dataset, evaluators.split(","))

    # Report results
    console.print(results.summary_table())
    results.save(output_dir)

@app.command()
def ci(
    threshold_file: str = typer.Option("./thresholds.yaml"),
):
    """Run CI regression suite with pass/fail gating"""

    results = run_regression_suite()
    thresholds = load_thresholds(threshold_file)

    passed = check_thresholds(results, thresholds)

    if not passed:
        console.print("[red]CI FAILED: Thresholds not met[/red]")
        raise typer.Exit(1)

    console.print("[green]CI PASSED[/green]")

if __name__ == "__main__":
    app()
```

---

### Phase 7: CI/CD Integration (Week 7-8)

#### 7.1 GitHub Actions - Regression
```yaml
# .github/workflows/ci-regression.yml
name: Eval Regression

on:
  pull_request:
    paths:
      - 'src/**'
      - 'promptfoo/**'

jobs:
  regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e .

      - name: Run Promptfoo regression
        env:
          NUVII_API_URL: ${{ secrets.NUVII_API_URL }}
          NUVII_API_KEY: ${{ secrets.NUVII_API_KEY }}
        run: |
          cd promptfoo
          npx promptfoo eval --config promptfooconfig.yaml

      - name: Run Python regression
        run: python scripts/run_eval.py ci

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: runs/
```

#### 7.2 GitHub Actions - Full Evaluation
```yaml
# .github/workflows/ci-eval.yml
name: Full Evaluation

on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday
  workflow_dispatch:

jobs:
  full-eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run full evaluation
        run: |
          python scripts/run_eval.py run \
            --dataset datasets/golden/all_test_cases.jsonl \
            --evaluators icd,cpt,em,hcc,gap,query,ragas \
            --output-dir runs/$(date +%Y%m%d)

      - name: Post to Slack
        if: always()
        run: python scripts/post_results_slack.py
```

---

## 3. KPI Definitions

### Primary KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| ICD Top-1 Accuracy | ≥ 85% | Exact match on primary code |
| ICD Top-3 Accuracy | ≥ 95% | Primary code in top 3 suggestions |
| CPT Accuracy | ≥ 90% | Correct procedure code |
| E/M Exact Match | ≥ 80% | Exact E/M level match |
| E/M Within-1 Level | ≥ 98% | Within 1 level of expected |
| HCC Recall | ≥ 90% | Captured expected HCCs |
| Gap Detection Precision | ≥ 85% | True gaps / detected gaps |
| Gap Detection Recall | ≥ 80% | True gaps / expected gaps |
| Query Quality Score | ≥ 4.0/5.0 | Rubric composite score |

### Secondary KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Context Precision (RAGAS) | ≥ 0.8 | Relevant chunks retrieved |
| Faithfulness (RAGAS) | ≥ 0.9 | Claims grounded in context |
| Citation Accuracy | ≥ 90% | Evidence spans match claims |
| Latency P95 | < 5s | 95th percentile response time |
| Cost per Case | < $0.50 | Token cost per evaluation |
| PHI Leakage | 0 | No PHI in outputs |

---

## 4. Dataset Requirements

### Gold Standard Dataset Structure
```jsonl
{"id": "icd_001", "clinical_note": "...", "expected_icd_codes": ["E11.9", "I10"], "specialty": "endocrine", "complexity": "moderate"}
{"id": "icd_002", "clinical_note": "...", "expected_icd_codes": ["I50.22"], "acceptable_codes": ["I50.2", "I50.20"], "specialty": "cardiology"}
```

### Minimum Dataset Sizes
- **ICD Test Cases**: 200+ (across specialties)
- **CPT Test Cases**: 100+ (across procedure types)
- **E/M Test Cases**: 150+ (all levels, settings)
- **HCC Test Cases**: 100+ (all major HCC categories)
- **Gap Test Cases**: 150+ (all gap types)
- **Query Test Cases**: 100+ (quality rubric validation)
- **Regression Subset**: 50 cases (fast CI suite)

---

## 5. Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Foundation | Week 1-2 | Project setup, API client, schemas |
| Phase 2: Core Evaluators | Week 2-3 | ICD, HCC, Gap, Query, E/M evaluators |
| Phase 3: RAGAS Integration | Week 3-4 | RAGAS wrapper, context metrics |
| Phase 4: Phoenix | Week 4-5 | Tracing, PHI redaction, artifacts |
| Phase 5: Promptfoo | Week 5-6 | CI config, test assertions, provider |
| Phase 6: CLI & Runners | Week 6-7 | Batch runner, reporters, CLI |
| Phase 7: CI/CD | Week 7-8 | GitHub Actions, dashboards |

**Total: 8 weeks to MVP**

---

## 6. Dependencies

### Python Packages
```
phoenix>=4.0
ragas>=0.1
langchain>=0.2
pydantic>=2.0
httpx>=0.27
typer>=0.12
rich>=13.0
pytest>=8.0
pandas>=2.0
```

### External Services
- Nuvii CDI API (V2 endpoints)
- Phoenix (self-hosted or Arize cloud)
- Optional: LLM for judge evaluations (Claude/GPT-4)

---

## 7. Next Steps

1. **Create repository**: `nuvii-cdi-eval`
2. **Set up project structure**: Poetry, pre-commit, CI skeleton
3. **Create initial dataset**: 50 gold-labeled test cases
4. **Implement ICD evaluator**: First evaluator as proof of concept
5. **Integrate Phoenix**: Basic tracing for debugging
6. **Add Promptfoo**: CI regression gate

---

## 8. Open Questions

1. **Dataset source**: Do we have labeled clinical notes, or need to create them?
2. **LLM judge**: Use Claude/GPT-4 for query quality, or rules-only initially?
3. **Phoenix hosting**: Self-host or use Arize cloud?
4. **PHI handling**: Use synthetic notes or redact real notes?
5. **Baseline**: What are current accuracy numbers to set thresholds?
