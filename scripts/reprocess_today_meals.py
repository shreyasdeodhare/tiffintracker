"""Operator command: defaults to dry-run; applies only today's unresolved reviews."""

import argparse
import json

from app import create_app
from app.application.reprocess_meals import reprocess_today


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Explicitly apply eligible decisions")
    args = parser.parse_args()
    app = create_app()
    with app.app_context():
        print(json.dumps(reprocess_today(dry_run=not args.apply), indent=2))


if __name__ == "__main__":
    main()
