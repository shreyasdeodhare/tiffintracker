"""Read-only verification that this landing page has not changed existing files."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).with_name("preservation-baseline.json")


def main():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    changed = []
    for name, expected in baseline["files"].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(name)
    routes = []
    for path in (ROOT / "app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if (
                        isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Attribute)
                        and decorator.func.attr in {"route", "get", "post"}
                    ):
                        routes.append(
                            {
                                "file": path.relative_to(ROOT).as_posix(),
                                "function": node.name,
                                "decorator": ast.unparse(decorator),
                            }
                        )

    def key(item):
        return item["file"], item["function"], item["decorator"]

    routes_match = sorted(routes, key=key) == sorted(baseline["routes"], key=key)
    print(
        json.dumps(
            {
                "baseline_commit": baseline["commit"],
                "existing_files_checked": len(baseline["files"]),
                "changed_or_missing": changed,
                "http_route_declarations": len(routes),
                "routes_unchanged": routes_match,
            },
            indent=2,
        )
    )
    if changed or not routes_match:
        raise SystemExit("Preservation check failed; stop and review before proceeding")


if __name__ == "__main__":
    main()
