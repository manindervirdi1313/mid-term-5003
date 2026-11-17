"""Keep files."""

import json
import re
from pathlib import Path

import gradio
import joblib
import sklearn

ROOT = Path(__file__).resolve().parents[1]


def test_saved_pipeline_and_metadata_are_included():
    assert (ROOT / "models" / "no_show.joblib").exists()
    assert (ROOT / "models" / "model_info.json").exists()
    assert hasattr(joblib.load(ROOT / "models" / "no_show.joblib"), "predict_proba")


def test_sklearn_versions_match_in_all_three_places():
    info = json.loads((ROOT / "models" / "model_info.json").read_text())
    requirements = (ROOT / "requirements.txt").read_text()
    assert info["scikit_learn_version"] == sklearn.__version__
    assert f"scikit-learn=={sklearn.__version__}" in requirements


def test_gradio_is_not_a_space_requirement():
    requirements = (ROOT / "requirements.txt").read_text().lower()
    assert not re.search(r"^gradio\s*[=<>]", requirements, re.MULTILINE)


def test_readme_sdk_version_equals_installed_gradio():
    readme = (ROOT / "README.md").read_text()
    match = re.search(r"^sdk_version:\s*(\S+)", readme, re.MULTILINE)
    assert match is not None
    assert match.group(1) == gradio.__version__


def test_readme_app_file_is_present():
    readme = (ROOT / "README.md").read_text()
    match = re.search(r"^app_file:\s*(\S+)", readme, re.MULTILINE)
    assert match and (ROOT / match.group(1)).is_file()


def test_all_required_reports_are_included():
    for filename in ("cv_results.csv", "metrics.json", "threshold_search.csv"):
        assert (ROOT / "reports" / filename).exists()


def test_no_gitignore_rule_excludes_saved_model():
    ignore = (ROOT / ".gitignore").read_text().splitlines()
    assert "models/" not in ignore
    assert "*.joblib" not in ignore


def test_environment_name_and_runnable_training_module():
    assert "name: midterm-no-shows" in (ROOT / "environment.yml").read_text()
    assert (ROOT / "noshows" / "train.py").exists()
