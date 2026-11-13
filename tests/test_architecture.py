"""Keep dependency boundaries enforceable as the code evolves."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1] / "app"


def test_domain_has_no_framework_or_persistence_dependencies():
    for path in (ROOT / "domain").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("flask", "sqlalchemy", "app.infrastructure", "app.application")
                ), (path, node.module)


def test_http_routes_do_not_issue_database_queries():
    files = [
        ROOT / name for name in ["routes.py", "auth.py", "whatsapp_routes.py", "operations.py"]
    ]
    files += list((ROOT / "presentation").glob("*.py"))
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"query", "execute", "scalars", "get_or_404"}, (
                    path,
                    node.lineno,
                )


def test_infrastructure_does_not_import_application_or_presentation():
    for path in (ROOT / "infrastructure").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("app.application", "app.presentation")
                ), (path, node.module)


def test_application_use_cases_do_not_issue_sql():
    for path in (ROOT / "application").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"query", "execute", "scalars"}, (path, node.lineno)
