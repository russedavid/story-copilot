import argparse
import json

from .store import Store


def main():
    p = argparse.ArgumentParser(
        description="Reconstruct game state and develop a source-backed Facilitator."
    )
    p.add_argument("--home", help="Private data directory (or STORY_DATA).")
    sub = p.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("path")
    imp.add_argument("--story", required=True)
    imp.add_argument("--title")
    imp.add_argument("--audio")
    imp.add_argument(
        "--split",
        choices=["train", "validation", "test", "development"],
        default="development",
    )
    sub.add_parser("list")
    sub.add_parser("example")
    ui = sub.add_parser("serve")
    ui.add_argument("--port", type=int, default=5022)
    st = sub.add_parser("state")
    st.add_argument("collection")
    st.add_argument("--before", type=int)
    ex = sub.add_parser("extract")
    ex.add_argument("collection")
    ex.add_argument("--start", type=int, default=1)
    ex.add_argument("--end", type=int, default=10)
    dr = sub.add_parser("draft")
    dr.add_argument("collection")
    dr.add_argument("--before", type=int, required=True)
    dr.add_argument("--player-input", default="")
    dr.add_argument("--direction", default="")
    exp = sub.add_parser("export")
    exp.add_argument("collection")
    exp.add_argument("output")
    exp.add_argument(
        "--include-screened",
        action="store_true",
        help="Include explicitly model-screened targets; provenance retains that distinction.",
    )
    ru = sub.add_parser("rules-import")
    ru.add_argument("pdf")
    ru.add_argument("--title", required=True)
    ru.add_argument("--edition", default="custom")
    rs = sub.add_parser("rules-search")
    rs.add_argument("query")
    args = p.parse_args()
    store = Store(args.home)
    if args.command == "import":
        from .corpus import import_transcript

        result = import_transcript(
            store,
            args.path,
            story=args.story,
            title=args.title,
            split=args.split,
            audio=args.audio,
        )
    elif args.command == "example":
        from .demo import ensure_demo
        from .campaigns import Campaigns

        result = ensure_demo(Campaigns(store))
    elif args.command == "list":
        result = store.collections()
    elif args.command == "serve":
        import uvicorn
        from .web import create_app

        uvicorn.run(create_app(store), host="127.0.0.1", port=args.port)
        return
    elif args.command == "state":
        from .state import replay

        result = replay(store, args.collection, before=args.before)
    elif args.command == "extract":
        from .model import extract

        rid = extract(store, args.collection, args.start, args.end)
        result = next(r for r in store.runs(args.collection) if r["id"] == rid)
    elif args.command == "draft":
        from .model import draft

        rid = draft(
            store, args.collection, args.before, args.player_input, args.direction
        )
        result = next(r for r in store.runs(args.collection) if r["id"] == rid)
    elif args.command == "export":
        from .training import export_dataset

        result = export_dataset(
            store, args.collection, args.output, include_screened=args.include_screened
        )
    elif args.command == "rules-import":
        from .rules import import_rules

        result = import_rules(store, args.pdf, args.title, args.edition)
    elif args.command == "rules-search":
        from .rules import search_rules

        result = search_rules(store, args.query)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
