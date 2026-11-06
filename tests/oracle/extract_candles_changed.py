"""Cut the candle replay for the Rust test out of the committed fixtures.

The Rust crate has no gzip dependency, so `src/quant/fns/candles.rs` replays this plain-JSON
extract instead of `talib_pattern_recognition.json.gz` and `talib_candles_changed.json.gz`:
the constructed `candles` series (the hex of each f64's big-endian bits, so nothing is
rounded), and for every pattern TA-Lib's lookback and its nonzero outputs under the default
settings and under the three changed ones.

    python3 tests/oracle/extract_candles_changed.py
"""

import gzip
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "python", "oracle", "fixtures")

def load(name):
    with gzip.open(os.path.join(FIX, name), "rt") as f:
        return json.load(f)


def ints(entry):
    return np.frombuffer(bytes.fromhex(entry["data"]), dtype=">i4").astype(np.int64)


def main():
    inputs = load("inputs.json.gz")["series"]["candles"]
    default = load("talib_pattern_recognition.json.gz")
    changed = load("talib_candles_changed.json.gz")
    cols = {k: inputs[k] for k in ("open", "high", "low", "close")}
    n = len(cols["open"]) // 16
    patterns = {}
    for name, f in default["functions"].items():
        a, b = f["cases"][0], changed["functions"][name]["cases"][0]
        assert a["scenario"] == b["scenario"] == "candles"
        da, db = ints(a["out"]["integer"])[:n], ints(b["out"]["integer"])[:n]
        patterns[name] = {
            "lookback_default": a["lookback"],
            "lookback_changed": b["lookback"],
            "default": [[int(i), int(da[i])] for i in np.flatnonzero(da)],
            "changed": [[int(i), int(db[i])] for i in np.flatnonzero(db)],
        }
    out = {
        "source": "talib_pattern_recognition.json.gz and talib_candles_changed.json.gz",
        "n": n,
        "settings_default": default["header"]["candle_settings"],
        "settings_changed": changed["header"]["candle_settings"],
        "columns": cols,
        "patterns": patterns,
    }
    with open(os.path.join(HERE, "candles_changed_extract.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"), sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
