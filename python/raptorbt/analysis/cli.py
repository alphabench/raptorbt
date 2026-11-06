"""``python -m raptorbt``: the analysis session as shell commands that print JSON.

Every command prints one JSON document on stdout. A refusal prints the error's ``to_dict()``
(``type``, ``code``, ``message``, ``operation``, ``parameter``, ``value``, ``valid``, ``skill``)
and exits 2; success exits 0. The one exception is ``skills show``, which prints the skill's
markdown.

A session lives under the cache directory (``$XDG_CACHE_HOME/raptorbt/sessions/<name>``, else
``~/.cache/raptorbt/sessions/<name>``), so a handle printed by one command is a series in the
next. Sessions unused for 7 days are deleted, and the oldest go first when the cache passes
1 GiB.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from raptorbt import errors, expressions, skills
from raptorbt.analysis import disk
from raptorbt.analysis.files import SUFFIXES, read_bars, read_panel
from raptorbt.analysis.session import Session, _plain

#: Lists longer than this in a ``stats`` result print as head and tail.
LIST_LIMIT = 20
_FIRST_SENTENCE = re.compile(r"(.+?[.;])(\s|$)", re.S)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise errors.InvalidParameter(
            message, code="invalid_parameter", operation="python -m raptorbt",
            valid=[self.format_usage().strip()],
        )


def _parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="python -m raptorbt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, parser_class=_Parser)

    def sessioned(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--session", default="default", help="session name (default: default)")
        p.add_argument("--cache-dir", help="replaces ~/.cache/raptorbt (sessions go in <dir>/sessions)")
        p.add_argument("--max-bytes", type=int, help="session memory budget in bytes (default 1 GiB)")
        return p

    def data_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", help="bars: a .csv, .npz or .npy file (loaded into the session; omit to reuse it)")
        p.add_argument("--bar-stamp", choices=["open", "close"], help="whether a timestamp starts or ends its bar (required with --data)")
        p.add_argument("--bar-ns", type=int, help="bar width in nanoseconds")
        p.add_argument("--name", help="name of the dataset (default d_<hash>)")
        p.add_argument("--time-unit", default="ns", help="unit of integer CSV timestamps: ns, us, ms, s (default ns)")
        p.add_argument("--field", default="close", help="column held by a plain 1-D .npy (default close)")
        p.add_argument("--timestamps", help="int64 nanoseconds .npy for a plain 1-D .npy")

    e = sessioned(sub.add_parser("eval", help="evaluate an expression; prints its summary and handle"))
    e.add_argument("expression")
    data_options(e)
    e.add_argument("--panel", help="a panel of many instruments: a long .csv (time, instrument, prices) or an .npz of 2-D arrays "
                   "(loaded into the session; needs --bar-stamp)")
    e.add_argument("--instruments", help="comma-separated instrument names to select, in order (names the columns of an .npz)")
    e.add_argument("--use", help="which loaded dataset to evaluate over (default: the newest)")
    e.add_argument("--tz-offset-ns", type=int, default=0, help="zone daily windows of at(...) are cut in")

    f = sub.add_parser("functions", help="search the expression functions")
    f.add_argument("query", nargs="*")
    f.add_argument("--limit", type=int, default=20)

    d = sub.add_parser("describe", help="one function: signature, warm-up, reference, conventions, example")
    d.add_argument("name")

    s = sessioned(sub.add_parser("show", help="a handle: its recipe and the last values of its series"))
    s.add_argument("handle")
    s.add_argument("--tail", type=int, default=20)
    s.add_argument("--instruments", help="panel results: comma-separated names (default: the names at both ends of the last cross-section)")
    s.add_argument("--all", action="store_true", help="every value")

    fa = sessioned(sub.add_parser("factor", help="score a panel result (a handle or an expression) against forward returns"))
    fa.add_argument("target")
    fa.add_argument("--horizons", default="1,5,20", help="forward horizons in bars (default 1,5,20)")
    fa.add_argument("--quantiles", type=int, default=5)
    fa.add_argument("--lag", type=int, default=1, help="bars between the signal and the start of its return (default 1)")
    fa.add_argument("--min-names", type=int, default=10)
    fa.add_argument("--use", help="which loaded panel (default: the newest)")

    x = sessioned(sub.add_parser("export", help="write a handle's series to .csv, .npy or .npz"))
    x.add_argument("handle")
    x.add_argument("--out", required=True)
    x.add_argument("--force", action="store_true", help="replace an existing file")

    t = sessioned(sub.add_parser("stats", help="run a raptorbt.stats function on handles"))
    t.add_argument("name")
    t.add_argument("handles", nargs="*")
    t.add_argument("--args", default="{}", help='keyword arguments as a JSON object, e.g. \'{"alpha": 0.01}\'')
    t.add_argument("--use", help="which loaded dataset (for audit_bars and impact_estimate)")

    k = sub.add_parser("skills", help="the skills an agent loads")
    ks = k.add_subparsers(dest="skills_cmd", required=True, parser_class=_Parser)
    ks.add_parser("for", help="the skills a task needs, best first, then what they depend on").add_argument("task", nargs="+")
    ks.add_parser("show", help="print a skill (markdown)").add_argument("name")
    ke = ks.add_parser("export", help="copy the skills to DIR as <DIR>/raptorbt-<domain>/SKILL.md")
    ke.add_argument("dir")
    ke.add_argument("--task", help="export only the skills this task needs, and what they depend on")
    return ap


# ------------------------------------------------------------------ helpers


def _open(args: argparse.Namespace):
    root = disk.cache_root(args.cache_dir)
    path = disk.session_dir(root, args.session)
    return root, path


def _compact(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _compact(v) for k, v in x.items()}
    if isinstance(x, list) and len(x) > LIST_LIMIT:
        return {"n": len(x), "head": _compact(x[:5]), "tail": _compact(x[-5:])}
    if isinstance(x, list):
        return [_compact(v) for v in x]
    return x


def _first_sentence(doc: str, limit: int = 140) -> str:
    m = _FIRST_SENTENCE.match(doc.strip())
    text = m.group(1) if m else doc.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _with_session(args: argparse.Namespace, work):
    """Open the named session under its lock, run ``work(session)``, save what changed."""
    root, path = _open(args)
    disk.purge(root, keep=path)
    with disk.locked(path):
        session = disk.open_session(path, max_bytes=args.max_bytes)
        out = work(session)
        disk.save(session, path)
        disk.touch(path)
    return out


# ------------------------------------------------------------------ commands


def _select_columns(bars: dict[str, Any], named: list[str], picked: list[str]) -> tuple[dict[str, Any], list[str]]:
    """The columns of 2-D arrays whose names (``named``) are in ``picked``, in that order."""
    index = {n: j for j, n in enumerate(named)}
    for n in picked:
        if n not in index:
            raise errors.UnknownSymbol(
                f"no instrument {n!r} in the panel", code="unknown_symbol", operation="eval",
                parameter="instruments", value=n, valid=named[:20],
            )
    cols = [index[n] for n in picked]
    return {k: (v[:, cols] if getattr(v, "ndim", 1) == 2 else v) for k, v in bars.items()}, picked


def _eval(a: argparse.Namespace) -> Any:
    def work(s: Session) -> Any:
        if a.data is not None:
            bars = read_bars(a.data, time_unit=a.time_unit, field=a.field, timestamps=a.timestamps)
            s.load(bars, bar_stamp=a.bar_stamp, bar_ns=a.bar_ns, name=a.name)
        if a.panel is not None:
            bars, named = read_panel(a.panel, time_unit=a.time_unit)
            picked = a.instruments.split(",") if a.instruments else named
            if named is not None and a.instruments:
                bars, picked = _select_columns(bars, named, picked)
            s.load_panel(bars, bar_stamp=a.bar_stamp, bar_ns=a.bar_ns, instruments=picked, name=a.name)
        return s.eval(a.expression, data=a.use, tz_offset_ns=a.tz_offset_ns)

    return _with_session(a, work)


def _functions(a: argparse.Namespace) -> Any:
    rows = expressions.functions(" ".join(a.query) or None)
    shown = [
        {"name": r["name"], "family": r["family"], "signature": r["signature"], "talib": r["talib"],
         "doc": _first_sentence(r["doc"])}
        for r in rows[: max(a.limit, 0)]
    ]
    return {"count": len(rows), "shown": len(shown), "functions": shown}


def _describe(a: argparse.Namespace) -> Any:
    return expressions.describe(a.name)


def _show(a: argparse.Namespace) -> Any:
    def work(s: Session) -> Any:
        info = s.describe_handle(a.handle)
        names = a.instruments.split(",") if a.instruments else (info["highlights"] or None)
        values = s.series(a.handle, None if a.all else a.tail, names)
        if isinstance(values, dict):
            values = {k: v for k, v in values.items()}
        return {**info, "tail": None if a.all else a.tail, "instruments": names, "values": values}

    return _plain(_with_session(a, work))


def _export(a: argparse.Namespace) -> Any:
    out = Path(a.out)
    if out.suffix.lower() not in SUFFIXES:
        raise errors.InvalidParameter(
            f"cannot write {out.name}: expected {', '.join(SUFFIXES)}", code="invalid_input",
            operation="export", parameter="out", value=a.out, valid=list(SUFFIXES),
        )
    if out.exists() and not a.force:
        raise errors.InvalidParameter(
            f"{out} exists; pass --force to replace it", code="invalid_input", operation="export",
            parameter="out", value=a.out, valid=["--force", "another path"],
        )

    def work(s: Session) -> Any:
        info = s.describe_handle(a.handle)
        got = s.series(a.handle)
        cols = got if isinstance(got, dict) else {a.handle: got}
        ds = s.dataset(info["data"])
        ts = ds.arrays["timestamps"]
        if out.suffix.lower() != ".npz" and any(c.ndim != 1 for c in cols.values()):
            raise errors.InvalidParameter(
                f"{a.handle} holds a panel (dates by names); write .npz, which keeps the shape",
                code="invalid_input", operation="export", parameter="out", value=a.out, valid=[".npz"],
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".csv":
            with open(out, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["timestamps", *cols])
                for i in range(len(ts)):
                    w.writerow([int(ts[i]), *("" if np.isnan(c[i]) else repr(float(c[i])) for c in cols.values())])
        elif out.suffix.lower() == ".npz":
            extra = {"instruments": np.array(ds.names)} if ds.kind == "panel" else {}
            np.savez(out, timestamps=ts, **extra, **cols)
        else:
            if len(cols) != 1:
                raise errors.InvalidParameter(
                    f"{a.handle} has {len(cols)} outputs and a .npy holds one series: name one "
                    f"({', '.join(info['outputs'])}) or write .npz", code="invalid_input",
                    operation="export", parameter="handle", value=a.handle, valid=info["outputs"],
                )
            np.save(out, next(iter(cols.values())), allow_pickle=False)
        return {"path": str(out), "rows": len(ts), "columns": ["timestamps", *cols] if out.suffix.lower() != ".npy" else [a.handle]}

    return _with_session(a, work)


def _factor(a: argparse.Namespace) -> Any:
    try:
        horizons = [int(x) for x in a.horizons.split(",")]
    except ValueError:
        raise errors.InvalidParameter(
            f"--horizons is comma-separated whole numbers, got {a.horizons!r}", code="invalid_parameter",
            operation="factor", parameter="horizons", value=a.horizons, valid=["1,5,20"],
        ) from None
    return _with_session(
        a, lambda s: s.factor(a.target, horizons=horizons, quantiles=a.quantiles, lag=a.lag,
                              min_names=a.min_names, data=a.use),
    )


def _stats(a: argparse.Namespace) -> Any:
    try:
        kw = json.loads(a.args)
    except json.JSONDecodeError as e:
        raise errors.InvalidParameter(
            f"--args is not JSON: {e}", code="invalid_input", operation="stats", parameter="args",
            value=a.args, valid=['a JSON object such as \'{"alpha": 0.01}\''],
        ) from None
    if not isinstance(kw, dict):
        raise errors.InvalidParameter(
            "--args must be a JSON object", code="invalid_input", operation="stats", parameter="args",
            value=a.args, valid=['a JSON object such as \'{"alpha": 0.01}\''],
        )
    return _compact(_plain(_with_session(a, lambda s: s.stats(a.name, *a.handles, data=a.use, **kw))))


def _skills(a: argparse.Namespace) -> Any:
    if a.skills_cmd == "for":
        by_name = {s.name: s for s in skills.index()}
        names = skills.for_task(" ".join(a.task))
        return {"skills": [{"name": n, "title": by_name[n].title, "depends_on": list(by_name[n].depends_on)} for n in names]}
    if a.skills_cmd == "show":
        return skills.load(a.name)
    return _export_skills(Path(a.dir), a.task)


_LINK = re.compile(r"\]\(\.\./([a-z-]+)/SKILL\.md")


def _export_skills(dest: Path, task: str | None) -> dict[str, Any]:
    infos = skills.index()
    wanted = None if task is None else set(skills.for_task(task))
    root = resources.files("raptorbt.skills").joinpath("_data")
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for s in infos:
        if wanted is not None and s.name not in wanted:
            continue
        src = root.joinpath(s.path).parent
        target = dest / s.name
        target.mkdir(exist_ok=True)
        for item in src.iterdir():
            if not item.is_file():
                continue
            text = item.read_text(encoding="utf-8")
            if item.name.endswith(".md"):
                text = _LINK.sub(lambda m: f"](../raptorbt-{m.group(1)}/SKILL.md", text)
            (target / item.name).write_text(text, encoding="utf-8")
        written.append(s.name)
    for name in ("INDEX.md", "manifest.json"):
        (dest / name).write_text(root.joinpath(name).read_text(encoding="utf-8"), encoding="utf-8")
    return {"dir": str(dest), "skills": written, "files": ["INDEX.md", "manifest.json"]}


_COMMANDS = {
    "eval": _eval, "functions": _functions, "describe": _describe, "show": _show,
    "export": _export, "stats": _stats, "factor": _factor, "skills": _skills,
}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = _COMMANDS[args.cmd](args)
    except errors.RaptorBTError as e:
        print(json.dumps(e.to_dict(), separators=(",", ":")))
        return 2
    except OSError as e:
        err = errors.InvalidData(
            f"{type(e).__name__}: {e}", code="invalid_input", operation="python -m raptorbt",
            valid=["a readable path in an existing directory"],
        )
        print(json.dumps(err.to_dict(), separators=(",", ":")))
        return 2
    if isinstance(result, str):
        sys.stdout.write(result)
    else:
        print(json.dumps(_plain(result), separators=(",", ":"), allow_nan=False))
    return 0
