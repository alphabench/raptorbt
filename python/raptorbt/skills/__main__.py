"""``python -m raptorbt.skills`` command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from raptorbt.skills import _check


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m raptorbt.skills", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="verify every skill against the installed package")
    c.add_argument(
        "--strict",
        action="store_true",
        help="also require every public module and symbol to be covered and no skill to be a draft",
    )
    c.add_argument("--root", type=Path, help="check another skills directory (default: the packaged one)")
    c.add_argument("--timeout", type=float, default=60.0, help="seconds allowed per code example")

    i = sub.add_parser("index", help="print the registry section of INDEX.md, rendered from the manifest")
    i.add_argument("--write", action="store_true", help="rewrite it in place in skills/INDEX.md")

    t = sub.add_parser("for-task", help="list the skills that match a task description")
    t.add_argument("text", nargs="+")

    args = ap.parse_args(argv)
    if args.cmd == "check":
        return _check.run(root=args.root, strict=args.strict, timeout=args.timeout, out=sys.stdout)
    if args.cmd == "index":
        return _check.print_or_write_index(write=args.write, out=sys.stdout)
    from raptorbt import skills

    for name in skills.for_task(" ".join(args.text)):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
