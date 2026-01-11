"""
CLI commands for Nuvii CDI Evaluation Framework.
"""

from nuvii_eval.cli.commands.compare import app as compare_app
from nuvii_eval.cli.commands.dataset import app as dataset_app
from nuvii_eval.cli.commands.report import app as report_app
from nuvii_eval.cli.commands.run import app as run_app

__all__ = [
    "run_app",
    "compare_app",
    "report_app",
    "dataset_app",
]
