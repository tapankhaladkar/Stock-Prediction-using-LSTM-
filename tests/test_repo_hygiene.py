"""Guards against the problems the original repo had: committed secrets, committed notebook
outputs, and notebooks that no longer parse."""
import json
import re
import subprocess
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".ipynb", ".md", ".txt", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".json", ".example"}
SKIP_DIRS = {".git", "data", ".venv", "venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
# e.g. api_key='<40 hex characters>', token = "<long random string>"
SECRET = re.compile(r"""(?i)(api[_-]?key|token|secret|password)\W{0,6}['"][0-9a-z_\-]{32,}['"]""")


def project_files():
    for p in ROOT.rglob("*"):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS) and (p.suffix in TEXT_SUFFIXES):
            yield p


def notebooks():
    return sorted(p for p in project_files() if p.suffix == ".ipynb")


def test_no_credentials_in_project_files():
    hits = [f"{p.relative_to(ROOT)}: {m.group(0)[:30]}..." for p in project_files()
            if p.name != "test_repo_hygiene.py"
            for m in SECRET.finditer(p.read_text(encoding="utf-8", errors="ignore"))]
    assert not hits, "possible committed credential(s):\n" + "\n".join(hits)


def test_the_leaked_key_pattern_would_be_caught():
    assert SECRET.search("df = pdr.get_data_tiingo('AAPL', api_key='" + "a1b2c3d4" * 5 + "')")
    assert not SECRET.search("api_key=os.environ['TIINGO_API_KEY']")


def test_env_files_and_data_are_git_ignored():
    ignore = (ROOT / ".gitignore").read_text()
    for entry in (".env", "data/"):
        assert entry in ignore.splitlines()


@pytest.mark.parametrize("nb_path", notebooks(), ids=lambda p: p.name)
def test_notebooks_are_valid_and_committed_without_outputs(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert cell.outputs == [], f"{nb_path.name} cell {i} has saved output (strip before committing)"
            assert cell.execution_count is None, f"{nb_path.name} cell {i} has an execution count"
            compile(cell.source, f"{nb_path.name}[cell {i}]", "exec")     # syntax check


def test_there_is_exactly_one_notebook_no_duplicates():
    assert len(notebooks()) == 1, [p.name for p in notebooks()]


def test_readme_tells_owners_to_revoke_the_leaked_key():
    """History still holds the old key (rewriting a public repo's history is a decision for the
    owner); the README must tell people to treat it as compromised."""
    readme = (ROOT / "README.md").read_text()
    assert "revoke" in readme.lower() and "history" in readme.lower()


def test_readme_image_links_point_to_files_that_exist():
    """A results section with broken images is worse than none."""
    readme = (ROOT / "README.md").read_text()
    links = re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", readme)
    assert links, "README is expected to embed the results figures"
    local = [link for link in links if not link.startswith(("http://", "https://"))]
    missing = [link for link in local if not (ROOT / link).is_file()]
    assert not missing, f"README images not found: {missing}"
