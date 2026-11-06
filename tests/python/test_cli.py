"""``python -m raptorbt``: JSON on stdout, exit 2 with the error's fields on a refusal, and a
session on disk that a shell agent can chain handles through.

Every test runs the real command in a subprocess with its own cache directory.
"""

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from raptorbt.analysis import disk

MIN = 60 * 10**9


def bars(n=200, seed=2):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return {
        "timestamps": np.arange(1, n + 1, dtype=np.int64) * MIN,
        "open": open_,
        "high": np.maximum(open_, close) + 0.3,
        "low": np.minimum(open_, close) - 0.3,
        "close": close,
        "volume": np.round(rng.uniform(500.0, 1500.0, n)),
    }


def write_csv(path, d, stamp_col="timestamp"):
    keys = [k for k in d if k != "timestamps"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([stamp_col, *keys])
        for i in range(len(d["timestamps"])):
            w.writerow([int(d["timestamps"][i]), *(repr(float(d[k][i])) for k in keys)])
    return path


class Cli:
    """Runs the command against one cache directory."""

    def __init__(self, tmp_path):
        self.tmp = Path(tmp_path)
        self.cache = self.tmp / "cache"

    def run(self, *args, cache=True, env=None):
        cmd = [sys.executable, "-m", "raptorbt", *args]
        if cache:
            cmd += ["--cache-dir", str(self.cache)]
        e = {**os.environ, "RAPTORBT_NO_VERSION_CHECK": "1", **(env or {})}
        done = subprocess.run(cmd, capture_output=True, text=True, env=e, cwd=self.tmp)
        return done

    def ok(self, *args, **kw):
        done = self.run(*args, **kw)
        assert done.returncode == 0, done.stdout + done.stderr
        return json.loads(done.stdout)

    def refused(self, *args, **kw):
        done = self.run(*args, **kw)
        assert done.returncode == 2, (done.returncode, done.stdout, done.stderr)
        return json.loads(done.stdout)


def files_under(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*"))


# ----------------------------------------------------------------------- chaining


def test_handles_chain_across_invocations_and_the_numbers_match_a_python_session(tmp_path):
    from raptorbt.analysis import Session

    cli = Cli(tmp_path)
    d = bars()
    csv_path = write_csv(tmp_path / "bars.csv", d)
    first = cli.ok("eval", "rsi(close, 14)", "--data", str(csv_path), "--bar-stamp", "close")
    second = cli.ok("eval", f"zscore({first['handle']}, 20)")  # no data: the session still holds it
    shown = cli.ok("show", second["handle"], "--tail", "3")
    assert first["warmup"] == 14 and second["warmup"] == 33 and first["kind"] == "number"
    assert shown["handle"] == second["handle"] and len(shown["values"]) == 3
    assert shown["depends_on"] == [first["handle"]] and "recipe" in shown

    s = Session()
    s.load(d, bar_stamp="close")
    a = s.eval("rsi(close, 14)")
    b = s.eval(f"zscore({a['handle']}, 20)")
    assert (a["handle"], b["handle"]) == (first["handle"], second["handle"])  # one handle everywhere
    assert a["sha256"] == first["sha256"] and b["sha256"] == second["sha256"]  # CSV floats round-trip
    assert shown["values"] == [float(x) for x in s.series(b["handle"], tail=3)]


def test_a_repeat_in_another_invocation_is_the_same_handle_and_a_chained_stats_call_runs(tmp_path):
    cli = Cli(tmp_path)
    write_csv(tmp_path / "bars.csv", bars(400))
    a = cli.ok("eval", "close - sma(close, 20)", "--data", str(tmp_path / "bars.csv"), "--bar-stamp", "close")
    assert cli.ok("eval", "close - sma(close, 20)")["handle"] == a["handle"]
    r = cli.ok("stats", "stationarity", a["handle"])
    assert r["analysis"] == {"handles": [a["handle"]], "trimmed_leading": 19} and "reading" in r
    r2 = cli.ok("stats", "hac_tstat", a["handle"], "--args", '{"lags": 3}')
    assert r2["lags"] == 3


def test_a_stats_result_with_long_lists_prints_their_head_and_tail(tmp_path):
    cli = Cli(tmp_path)
    write_csv(tmp_path / "bars.csv", bars(600))
    h = cli.ok("eval", "log_return(close)", "--data", str(tmp_path / "bars.csv"), "--bar-stamp", "close")["handle"]
    r = cli.ok("stats", "markov_regimes", h, "--args", '{"series": true}')
    cut = [v for v in r.values() if isinstance(v, dict) and set(v) == {"n", "head", "tail"}]
    assert cut and all(c["n"] > 20 and len(c["head"]) == len(c["tail"]) == 5 for c in cut)
    assert len(json.dumps(r)) < 20_000


def test_export_writes_csv_npz_and_npy_and_refuses_to_overwrite(tmp_path):
    cli = Cli(tmp_path)
    write_csv(tmp_path / "bars.csv", bars())
    h = cli.ok("eval", "sma(close, 5)", "--data", str(tmp_path / "bars.csv"), "--bar-stamp", "close")["handle"]
    out = cli.ok("export", h, "--out", str(tmp_path / "x.csv"))
    assert out["rows"] == 200
    with open(tmp_path / "x.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["timestamps", h] and rows[1][1] == "" and rows[5][1] != ""  # unknown is an empty cell
    cli.ok("export", h, "--out", str(tmp_path / "x.npy"))
    assert np.load(tmp_path / "x.npy").shape == (200,)
    cli.ok("export", h, "--out", str(tmp_path / "x.npz"))
    assert set(np.load(tmp_path / "x.npz").files) == {"timestamps", h}
    err = cli.refused("export", h, "--out", str(tmp_path / "x.csv"))
    assert err["code"] == "invalid_input" and "--force" in err["valid"]
    cli.ok("export", h, "--out", str(tmp_path / "x.csv"), "--force")
    macd = cli.ok("eval", "macd(close)")
    err = cli.refused("export", macd["handle"], "--out", str(tmp_path / "m.npy"))
    assert err["code"] == "invalid_input"
    cli.ok("export", macd["handle"], "--out", str(tmp_path / "m.csv"))
    with open(tmp_path / "m.csv", newline="") as fh:
        assert next(csv.reader(fh))[0] == "timestamps"


def test_an_evicted_handle_prints_its_recipe_and_exits_two(tmp_path):
    cli = Cli(tmp_path)
    write_csv(tmp_path / "bars.csv", bars())
    n = 200 * 8
    budget = 6 * n + 2 * n  # six columns and two series
    a = cli.ok("eval", "sma(close, 3)", "--data", str(tmp_path / "bars.csv"), "--bar-stamp", "close",
               "--max-bytes", str(budget))
    cli.ok("eval", "sma(close, 4)")
    cli.ok("eval", "sma(close, 5)")  # the budget (kept in the session) holds two: a goes
    err = cli.refused("show", a["handle"])
    assert err["code"] == "handle_expired" and err["valid"][0].startswith('Session.eval("sma(close, 3)"')
    assert cli.ok("eval", "sma(close, 3)")["handle"] == a["handle"]  # the recipe works


def test_no_command_changes_the_input_file(tmp_path):
    cli = Cli(tmp_path)
    p = write_csv(tmp_path / "bars.csv", bars())
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    h = cli.ok("eval", "rsi(close, 14)", "--data", str(p), "--bar-stamp", "close")["handle"]
    cli.ok("show", h)
    cli.ok("export", h, "--out", str(tmp_path / "o.csv"))
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before


def test_the_summary_that_the_command_prints_fits_its_caps(tmp_path):
    cli = Cli(tmp_path)
    write_csv(tmp_path / "bars.csv", bars())
    first = cli.run("eval", "macd(close)", "--data", str(tmp_path / "bars.csv"), "--bar-stamp", "close")
    assert first.returncode == 0 and len(first.stdout.strip().encode()) <= 4096
    one = cli.run("eval", "rsi(close, 14)")
    assert len(one.stdout.strip().encode()) <= 2048


# ---------------------------------------------------------------------------- the data


def test_npz_npy_and_iso_timestamps_load(tmp_path):
    cli = Cli(tmp_path)
    d = bars(60)
    np.savez(tmp_path / "b.npz", **d)
    npz = cli.ok("eval", "sma(close, 5)", "--data", str(tmp_path / "b.npz"), "--bar-stamp", "close", "--session", "z")
    rec = np.rec.fromarrays([d[k] for k in d], names=list(d))
    np.save(tmp_path / "b.npy", rec)
    npy = cli.ok("eval", "sma(close, 5)", "--data", str(tmp_path / "b.npy"), "--bar-stamp", "close", "--session", "y")
    np.save(tmp_path / "c.npy", d["close"])
    np.save(tmp_path / "t.npy", d["timestamps"])
    plain = cli.ok("eval", "sma(close, 5)", "--data", str(tmp_path / "c.npy"), "--timestamps", str(tmp_path / "t.npy"),
                   "--bar-stamp", "close", "--session", "x")
    assert npz["handle"] != plain["handle"]  # plain has only a close column: other bars
    assert npz["sha256"] == npy["sha256"] == plain["sha256"]  # the same numbers
    iso = tmp_path / "iso.csv"
    with open(iso, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "close"])
        for i in range(40):
            w.writerow([f"2026-01-05T09:{i:02d}:00+05:30" if i < 60 else "", repr(100.0 + i)])
    ok = cli.ok("eval", "close", "--data", str(iso), "--bar-stamp", "close", "--session", "iso")
    assert ok["shape"] == [40]
    naive = tmp_path / "naive.csv"
    naive.write_text("time,close\n2026-01-05T09:00:00,1\n2026-01-05T09:01:00,2\n")
    err = cli.refused("eval", "close", "--data", str(naive), "--bar-stamp", "close", "--session", "n")
    assert err["code"] == "invalid_input" and "offset" in err["message"]


def test_csv_time_units_and_empty_cells(tmp_path):
    cli = Cli(tmp_path)
    p = tmp_path / "s.csv"
    p.write_text("ts,close\n" + "\n".join(f"{i},{100 + i}" if i != 5 else f"{i}," for i in range(1, 41)) + "\n")
    r = cli.ok("eval", "close", "--data", str(p), "--bar-stamp", "close", "--time-unit", "s")
    assert r["nan"] == {"leading": 0, "interior": 1}
    first = cli.ok("show", r["handle"], "--all")["values"]
    assert first[4] is None and first[5] == 106.0


def test_unreadable_data_is_a_typed_refusal(tmp_path):
    cli = Cli(tmp_path)
    for args, code in (
        (["--data", str(tmp_path / "missing.csv"), "--bar-stamp", "close"], "invalid_input"),
        (["--data", str(tmp_path / "x.txt"), "--bar-stamp", "close"], "invalid_input"),
        (["--data", str(tmp_path / "nostamp.csv")], "missing_bar_stamp"),
    ):
        (tmp_path / "x.txt").write_text("a")
        (tmp_path / "nostamp.csv").write_text("timestamp,close\n1,1\n2,2\n")
        err = cli.refused("eval", "close", *args)
        assert err["code"] == code, err


# -------------------------------------------------------------------- errors and exits


def test_a_bad_expression_prints_the_error_fields_and_exits_two(tmp_path):
    cli = Cli(tmp_path)
    write_csv(tmp_path / "bars.csv", bars(60))
    done = cli.run("eval", "1 + smaa(close, 5)", "--data", str(tmp_path / "bars.csv"), "--bar-stamp", "close")
    assert done.returncode == 2 and done.stderr == ""
    err = json.loads(done.stdout)
    assert err["type"] == "ExpressionError" and err["code"] == "expr_unknown_function"
    assert err["position"] == 4 and "sma" in err["valid"] and err["snippet"].endswith("^")
    assert err["skill"] == "expressions" and err["parameter"] == "expression" and err["value"] == "1 + smaa(close, 5)"
    assert set(err) >= {"type", "code", "message", "operation", "parameter", "value", "valid", "skill"}


def test_unknown_and_malformed_handles_and_usage_errors_exit_two_with_json(tmp_path):
    cli = Cli(tmp_path)
    write_csv(tmp_path / "bars.csv", bars(60))
    h = cli.ok("eval", "sma(close, 3)", "--data", str(tmp_path / "bars.csv"), "--bar-stamp", "close")["handle"]
    err = cli.refused("eval", "sma(h_0000000000, 3)")
    assert err["code"] == "unknown_handle" and h in err["valid"] + [h]
    assert cli.refused("show", "nonsense")["code"] == "unknown_handle"
    assert cli.refused("stats", "event_study", h)["code"] == "invalid_parameter"
    assert cli.refused("stats", "stationarity", h, "--args", "{not json")["code"] == "invalid_input"
    assert cli.refused("nosuchcommand", cache=False)["code"] == "invalid_parameter"
    assert cli.refused("eval", cache=False)["code"] == "invalid_parameter"  # missing argument
    assert cli.refused("describe", "smaa", cache=False)["code"] == "expr_unknown_function"
    assert cli.refused("eval", "close", "--bar-stamp", "start", cache=False)["code"] == "invalid_parameter"


def test_eval_without_data_in_a_new_session_says_what_is_missing(tmp_path):
    err = Cli(tmp_path).refused("eval", "close", "--session", "empty")
    assert err["code"] == "missing_field" and err["parameter"] == "data"


# ----------------------------------------------------------------- the cache and its paths


def test_a_session_or_data_name_with_a_path_cannot_leave_the_cache(tmp_path):
    cli = Cli(tmp_path)
    write_csv(tmp_path / "bars.csv", bars(60))
    before = files_under(tmp_path)
    for bad in ("../evil", "..", ".", "a/b", "/abs", "x\\y", "", ".hidden", "a" * 65):
        err = cli.refused("eval", "close", "--data", str(tmp_path / "bars.csv"), "--bar-stamp", "close", "--session", bad)
        assert err["code"] == "invalid_parameter", (bad, err)
    assert files_under(tmp_path) == before  # a bad session name creates nothing, here or beside the cache
    err = cli.refused("eval", "close", "--data", str(tmp_path / "bars.csv"), "--bar-stamp", "close", "--name", "../evil")
    assert err["code"] == "invalid_parameter"
    outside = [p for p in files_under(tmp_path) if not p.startswith("cache")]
    assert outside == [p for p in before if not p.startswith("cache")]
    assert not (tmp_path.parent / "evil").exists() and not (cli.cache / "evil").exists()


def test_a_handle_name_with_separators_never_becomes_a_path(tmp_path):
    cli = Cli(tmp_path)
    write_csv(tmp_path / "bars.csv", bars(60))
    h = cli.ok("eval", "sma(close, 3)", "--data", str(tmp_path / "bars.csv"), "--bar-stamp", "close")["handle"]
    secret = tmp_path / "secret.npy"
    np.save(secret, np.arange(5.0))
    before = files_under(tmp_path)
    for bad in (f"../../{secret.name}", f"{h}/../../..", "h_0123456789/../x", f"{h}_../../secret", "h_0123456789_a/b", "../" + h):
        for cmd in (["show", bad], ["export", bad, "--out", str(tmp_path / "out.csv")],
                    ["eval", f"sma({bad}, 3)"], ["stats", "stationarity", bad]):
            done = cli.run(*cmd)
            assert done.returncode == 2, (cmd, done.stdout)
            assert json.loads(done.stdout)["code"] in ("unknown_handle", "expr_syntax", "expr_type", "expr_unknown_function"), done.stdout
    assert not (tmp_path / "out.csv").exists()
    assert files_under(tmp_path) == before


def test_a_session_file_cannot_be_forged_into_reading_outside_the_cache(tmp_path):
    cli = Cli(tmp_path)
    write_csv(tmp_path / "bars.csv", bars(60))
    cli.ok("eval", "sma(close, 3)", "--data", str(tmp_path / "bars.csv"), "--bar-stamp", "close")
    meta = cli.cache / "sessions" / "default" / "meta.json"
    m = json.loads(meta.read_text())
    m["handles"][0]["id"] = "../../../etc/passwd"
    meta.write_text(json.dumps(m))
    err = cli.refused("show", "h_0123456789")
    assert err["code"] == "invalid_array"
    m = json.loads(meta.read_text())
    m["handles"][0]["id"] = "h_0123456789"
    m["data"] = {"../x": m["data"][next(iter(m["data"]))]}
    meta.write_text(json.dumps(m))
    assert cli.refused("show", "h_0123456789")["code"] == "invalid_array"


def test_the_default_cache_follows_xdg_cache_home(tmp_path):
    cli = Cli(tmp_path)
    write_csv(tmp_path / "bars.csv", bars(60))
    xdg = tmp_path / "xdg"
    cli.ok("eval", "close", "--data", str(tmp_path / "bars.csv"), "--bar-stamp", "close", cache=False,
           env={"XDG_CACHE_HOME": str(xdg)})
    assert (xdg / "raptorbt" / "sessions" / "default" / "meta.json").is_file()
    assert disk.cache_root() == (Path(os.environ["XDG_CACHE_HOME"]) if "XDG_CACHE_HOME" in os.environ else Path.home() / ".cache") / "raptorbt"


def test_sessions_unused_for_seven_days_are_purged_and_the_oldest_go_past_the_cap(tmp_path):
    cli = Cli(tmp_path)
    write_csv(tmp_path / "bars.csv", bars(60))
    for name in ("old", "mid", "new"):
        cli.ok("eval", "close", "--data", str(tmp_path / "bars.csv"), "--bar-stamp", "close", "--session", name)
    sessions = cli.cache / "sessions"
    now = time.time()
    for name, age_days in (("old", 8), ("mid", 3), ("new", 0)):
        t = now - age_days * 86400
        os.utime(sessions / name / "meta.json", (t, t))
    cli.ok("eval", "close", "--session", "mid")  # any command purges first
    assert sorted(p.name for p in sessions.iterdir()) == ["mid", "new"]
    root = cli.cache
    size = sum(f.stat().st_size for f in (sessions / "mid").rglob("*") if f.is_file())
    removed = disk.purge(root, keep=sessions / "new", max_bytes=size + 1)
    assert removed == ["mid"] and [p.name for p in sessions.iterdir()] == ["new"]
    # the session in use is never purged, however old
    t = now - 30 * 86400
    os.utime(sessions / "new" / "meta.json", (t, t))
    assert disk.purge(root, keep=sessions / "new") == [] and (sessions / "new").is_dir()


def test_two_commands_at_once_on_one_session_do_not_corrupt_it(tmp_path):
    cli = Cli(tmp_path)
    write_csv(tmp_path / "bars.csv", bars(300))
    cli.ok("eval", "close", "--data", str(tmp_path / "bars.csv"), "--bar-stamp", "close")
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "raptorbt", "eval", f"sma(close, {p})", "--cache-dir", str(cli.cache)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path,
            env={**os.environ, "RAPTORBT_NO_VERSION_CHECK": "1"},
        )
        for p in range(3, 9)
    ]
    outs = [p.communicate() for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    handles = [json.loads(o)["handle"] for o, _ in outs]
    shown = [cli.ok("show", h, "--tail", "1")["handle"] for h in handles]  # every one was kept
    assert shown == handles


# ------------------------------------------------------------------ functions, describe, skills


def test_functions_and_describe_print_a_small_answer(tmp_path):
    cli = Cli(tmp_path)
    f = cli.ok("functions", "relative", "strength", cache=False)
    assert f["functions"][0]["name"] == "rsi" and f["count"] >= 1 and f["shown"] == len(f["functions"])
    assert set(f["functions"][0]) == {"name", "family", "signature", "talib", "doc"}
    assert len(f["functions"][0]["doc"]) <= 140
    many = cli.ok("functions", cache=False)
    assert many["shown"] == 20 and many["count"] > 150
    assert cli.ok("functions", "doji", "--limit", "2", cache=False)["shown"] == 2
    d = cli.ok("describe", "rsi", cache=False)
    assert d["example"] == "rsi(close, 14)" and d["warmup"] == 14 and "conventions" in d


def test_skills_for_show_and_export(tmp_path):
    cli = Cli(tmp_path)
    r = cli.ok("skills", "for", "explore", "a", "series", "and", "chain", "results", "before", "any", "strategy", "exists", cache=False)
    assert r["skills"][0]["name"] == "raptorbt-analysis"
    names = [s["name"] for s in r["skills"]]
    assert "raptorbt-core" in names and "raptorbt-expressions" in names  # its dependencies follow
    shown = cli.run("skills", "show", "analysis", cache=False)
    assert shown.returncode == 0 and shown.stdout.startswith("---\nname: raptorbt-analysis")
    out = tmp_path / "skills_out"
    done = cli.ok("skills", "export", str(out), cache=False)
    assert "raptorbt-analysis" in done["skills"] and len(done["skills"]) == 20
    text = (out / "raptorbt-analysis" / "SKILL.md").read_text()
    assert "(../raptorbt-expressions/SKILL.md)" in text and "(../expressions/SKILL.md)" not in text
    assert (out / "raptorbt-expressions" / "functions.md").is_file() and (out / "INDEX.md").is_file()
    sub = tmp_path / "some"
    part = cli.ok("skills", "export", str(sub), "--task", "garch volatility", cache=False)
    import raptorbt.skills as skills

    assert set(part["skills"]) == set(skills.for_task("garch volatility"))
    assert "raptorbt-statistics" in part["skills"] and "raptorbt-orders" not in part["skills"]
    assert sorted(p.name for p in sub.iterdir() if p.is_dir()) == sorted(part["skills"])
    assert cli.refused("skills", "show", "nope", cache=False)["code"] == "unknown_skill"


# ------------------------------------------------------------------------------- panels


def panel_csv(path, n=150, m=24, seed=4, late=(3, 30)):
    """A long CSV (time, instrument, close) of ``m`` names with a drift each; name ``late[0]``
    lists ``late[1]`` bars late."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 0.001, m)
    close = 100.0 * np.exp(np.cumsum(mu + rng.normal(0.0, 0.01, (n, m)), axis=0))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "instrument", "close"])
        for j in range(m):
            for i in range(n):
                if j == late[0] and i < late[1]:
                    continue
                w.writerow([(i + 1) * 86_400 * 10**9, f"N{j:02d}", repr(float(close[i, j]))])
    return close


def test_panel_handles_chain_across_invocations_and_the_factor_matches_a_session(tmp_path):
    from raptorbt.analysis import Session
    from raptorbt.analysis.files import read_panel

    cli = Cli(tmp_path)
    panel_csv(tmp_path / "p.csv")
    first = cli.ok("eval", "rank(roc(close, 20))", "--panel", str(tmp_path / "p.csv"), "--bar-stamp", "close")
    second = cli.ok("eval", f"zscore_xs({first['handle']})")  # the panel is still in the session
    assert first["shape"] == [150, 24] and first["coverage"]["min"] == 23 and second["warmup"] == 20
    assert {n for n, _ in first["last"]["top"]} <= {f"N{j:02d}" for j in range(24)}
    f = cli.ok("factor", second["handle"], "--horizons", "1,5", "--quantiles", "4")
    assert f["no_costs"] is True and f["quantiles"] == 4 and set(f["horizons"]) == {"1", "5"}
    again = cli.ok("factor", second["handle"], "--horizons", "1,5", "--quantiles", "4")
    assert again == f  # served from the stored report

    bars, _ = read_panel(tmp_path / "p.csv")
    s = Session()
    s.load_panel(bars, bar_stamp="close")
    a = s.eval("rank(roc(close, 20))")
    assert a["handle"] == first["handle"] and a["sha256"] == first["sha256"]
    g = s.factor(f"zscore_xs({a['handle']})", horizons=(1, 5), quantiles=4)
    assert g == f

    shown = cli.ok("show", second["handle"], "--tail", "2")
    assert shown["shape"] == [150, 24] and shown["instruments"] and len(shown["values"]) == 2
    assert len(shown["values"][0]) == len(shown["instruments"])
    pick = cli.ok("show", second["handle"], "--tail", "1", "--instruments", "N05,N01")
    assert pick["instruments"] == ["N05", "N01"] and len(pick["values"][0]) == 2
    rep = cli.ok("show", f["handle"], "--tail", "3")
    assert "ic_5" in str(rep["outputs"]) and len(rep["values"]["ic_5"]) == 3


def test_an_unknown_instrument_exits_two_with_the_nearest_names(tmp_path):
    cli = Cli(tmp_path)
    panel_csv(tmp_path / "p.csv")
    h = cli.ok("eval", "zscore_xs(close)", "--panel", str(tmp_path / "p.csv"), "--bar-stamp", "close")["handle"]
    err = cli.refused("show", h, "--instruments", "N5")
    assert err["code"] == "unknown_symbol" and err["parameter"] == "instruments" and err["value"] == "N5"
    assert "N05" in err["valid"] or err["valid"]
    err = cli.refused("eval", "close", "--panel", str(tmp_path / "p.csv"), "--bar-stamp", "close",
                      "--instruments", "N00,NOPE", "--session", "two")
    assert err["code"] == "unknown_symbol" and err["value"] == "NOPE"
    assert cli.refused("factor", "zscore_xs(close)", "--horizons", "1,x")["code"] == "invalid_parameter"
    assert cli.refused("factor", "zscore_xs(close)", "--session", "none")["code"] == "missing_field"


def test_a_panel_npz_loads_with_names_and_exports_back_as_npz_only(tmp_path):
    cli = Cli(tmp_path)
    close = panel_csv(tmp_path / "p.csv", late=(0, 0))
    names = np.array([f"Q{j}" for j in range(close.shape[1])])
    np.savez(tmp_path / "p.npz", close=close, timestamps=(np.arange(1, 151) * 86_400 * 10**9).astype(np.int64), instruments=names)
    r = cli.ok("eval", "rank(close)", "--panel", str(tmp_path / "p.npz"), "--bar-stamp", "close")
    assert r["shape"] == [150, 24] and r["last"]["top"][0][0].startswith("Q")
    cli.ok("export", r["handle"], "--out", str(tmp_path / "o.npz"))
    z = np.load(tmp_path / "o.npz")
    assert z[r["handle"]].shape == (150, 24) and list(z["instruments"]) == list(names)
    err = cli.refused("export", r["handle"], "--out", str(tmp_path / "o.csv"))
    assert err["code"] == "invalid_input" and err["valid"] == [".npz"]
    sel = cli.ok("eval", "rank(close)", "--panel", str(tmp_path / "p.npz"), "--bar-stamp", "close",
                 "--instruments", "Q3,Q1", "--session", "sel")
    assert sel["shape"] == [150, 2]


def test_a_panel_summary_from_the_command_fits_its_cap(tmp_path):
    cli = Cli(tmp_path)
    panel_csv(tmp_path / "p.csv")
    out = cli.run("eval", "macd(close)", "--panel", str(tmp_path / "p.csv"), "--bar-stamp", "close")
    assert out.returncode == 0 and len(out.stdout.strip().encode()) <= 4096
    f = cli.run("factor", "zscore_xs(roc(close, 5))")
    assert f.returncode == 0 and len(f.stdout.strip().encode()) <= 2048
