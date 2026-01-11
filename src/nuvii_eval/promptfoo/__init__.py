"""
Promptfoo CI integration for CDI evaluation.

This module provides tools for integrating CDI evaluation with Promptfoo
for continuous integration and regression testing.

Key components:
- Configuration generator for Promptfoo YAML
- Test case converters for Promptfoo format
- Assertion builders for CDI-specific metrics
- CI runner and result parser
- Regression detection utilities
"""

from nuvii_eval.promptfoo.assertions import (
    AssertionBuilder,
    EMAssertionBuilder,
    GapAssertionBuilder,
    HCCAssertionBuilder,
    ICDAssertionBuilder,
    QueryAssertionBuilder,
    get_assertion_builder,
)
from nuvii_eval.promptfoo.config import (
    ConfigGeneratorOptions,
    PromptfooAssertion,
    PromptfooConfig,
    PromptfooProvider,
    PromptfooTest,
    create_nuvii_provider,
    create_openai_provider,
    generate_em_config,
    generate_gap_config,
    generate_hcc_config,
    generate_icd_config,
    generate_promptfoo_config,
    generate_query_config,
)
from nuvii_eval.promptfoo.converter import (
    TestCaseConverter,
    convert_single_test,
    convert_test_suite,
    export_to_yaml,
)
from nuvii_eval.promptfoo.regression import (
    DetectorConfig,
    Regression,
    RegressionDetector,
    RegressionReport,
    RegressionSeverity,
    RegressionType,
    check_for_blockers,
    compare_results,
    get_regression_summary,
)
from nuvii_eval.promptfoo.runner import (
    AssertionResult,
    PromptfooResult,
    PromptfooRunner,
    RunConfig,
    TestResult,
    analyze_results,
    format_ci_report,
)

__all__ = [
    # Assertions
    "AssertionBuilder",
    "EMAssertionBuilder",
    "GapAssertionBuilder",
    "HCCAssertionBuilder",
    "ICDAssertionBuilder",
    "QueryAssertionBuilder",
    "get_assertion_builder",
    # Config
    "ConfigGeneratorOptions",
    "PromptfooAssertion",
    "PromptfooConfig",
    "PromptfooProvider",
    "PromptfooTest",
    "create_nuvii_provider",
    "create_openai_provider",
    "generate_em_config",
    "generate_gap_config",
    "generate_hcc_config",
    "generate_icd_config",
    "generate_promptfoo_config",
    "generate_query_config",
    # Converter
    "TestCaseConverter",
    "convert_single_test",
    "convert_test_suite",
    "export_to_yaml",
    # Regression
    "DetectorConfig",
    "Regression",
    "RegressionDetector",
    "RegressionReport",
    "RegressionSeverity",
    "RegressionType",
    "check_for_blockers",
    "compare_results",
    "get_regression_summary",
    # Runner
    "AssertionResult",
    "PromptfooResult",
    "PromptfooRunner",
    "RunConfig",
    "TestResult",
    "analyze_results",
    "format_ci_report",
]
