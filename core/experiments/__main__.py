"""python -m core.experiments: config-driven runs and local history."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "compare"):
        p = sub.add_parser(name)
        p.add_argument("config")
    for name in ("list", "inspect", "report", "rebuild", "import", "compare-saved"):
        p = sub.add_parser(name)
        p.add_argument("--store", default="outputs/experiments")
        if name in ("inspect", "report"):
            p.add_argument("id")
        elif name == "import":
            p.add_argument("source")
        elif name == "compare-saved":
            p.add_argument("ids", nargs="+")
        elif name == "list":
            p.add_argument("--status")
            p.add_argument("--tag")
    args = parser.parse_args()
    if args.command == "validate":
        from core.experiments.runner import validate_experiment

        result = validate_experiment(args.config).model_dump(mode="json")
    elif args.command == "run":
        from core.experiments.runner import run

        result = str(run(args.config))
    elif args.command == "compare":
        from core.experiments.comparison import compare

        result = str(compare(args.config))
    elif args.command == "report":
        from core.experiments.reporting import regenerate

        result = str(regenerate(args.store, args.id))
    elif args.command == "import":
        from core.experiments.migration import import_run

        result = str(import_run(args.source, args.store))
    elif args.command == "compare-saved":
        from core.experiments.comparison import compare_saved

        result = str(compare_saved(args.store, args.ids))
    else:
        from core.experiments.storage import RunStore

        store = RunStore(args.store)
        if args.command == "list":
            result = store.list(status=args.status, tag=args.tag)
        elif args.command == "inspect":
            result = store.inspect(args.id)
        else:
            result = store.rebuild()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
