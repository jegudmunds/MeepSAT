import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"


def _notebook_source(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return notebook, "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_notebooks_are_valid_json_without_saved_exceptions():
    for path in EXAMPLES.glob("*.ipynb"):
        notebook, _ = _notebook_source(path)
        errors = [
            output
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert not errors, f"{path.name} contains a saved exception"


def test_runnable_notebooks_use_explicit_dft_field_extraction():
    for path in EXAMPLES.glob("*.ipynb"):
        _, source = _notebook_source(path)
        if "simulation.run(" not in source:
            continue
        assert "'method': 'dft'" in source, path.name
        assert "stepfunctions.setup_dft_fields(" in source, path.name
        assert "stepfunctions.save_dft_fields" in source, path.name
        assert "stepfunctions.accumulate_efield_and_hfield" not in source, path.name
        assert "stepfunctions.save_accumulated_fields" not in source, path.name


def test_examples_use_correct_auxiliary_data_spelling():
    text_files = [ROOT / "README.md"]
    text_files.extend(EXAMPLES.rglob("*.py"))
    text_files.extend(EXAMPLES.rglob("*.sh"))
    text_files.extend(EXAMPLES.rglob("*.md"))
    text_files.extend(EXAMPLES.glob("*.ipynb"))

    for path in text_files:
        assert "auxilary_data" not in path.read_text(encoding="utf-8"), path


def test_examples_do_not_import_missing_legacy_stepfunctions_module():
    for path in EXAMPLES.rglob("*.py"):
        assert "stepfunctions_old" not in path.read_text(encoding="utf-8"), path
