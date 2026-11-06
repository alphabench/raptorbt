# Oracle fixtures

`generate.py` runs TA-Lib, and (`--families stats`) statsmodels, arch and scipy, and writes the
numbers into `tests/python/oracle/fixtures/*.json.gz` and `fixtures/stats/`. The tests replay
those files; no reference library is needed at test time. The generator never imports raptorbt.

Regenerate (group `oracle` in `pyproject.toml` pins the versions; the quick form):

    uv run --no-project --python 3.12 --with ta-lib==0.6.4 --with numpy python tests/oracle/generate.py

Inputs are drawn by the SplitMix64 in `generate.py`; each fixture header records library
versions, the input digest and the candle settings. `measure_legacy.py` prints how today's
array and streaming indicators differ from the fixtures (it changes nothing).

## Banned reference sources

Never read, port, translate or copy numbers from the source or output-derived code of:

- GPL packages: urca, tseries, rugarch, MSwM (and anything else under the GPL or AGPL)
- GSL (GNU Scientific Library)
- tulipindicators (LGPL)
- Numerical Recipes (its licence forbids use of the code)

A reference is a published formula or paper, or the output of a permissively licensed
package credited in `NOTICE` (TA-Lib BSD, statsmodels and scipy BSD-3, arch NCSA). Running
a permissive package to produce fixtures is allowed; porting its source line by line is
not: kernels are written from the specification, then checked against its numbers.

## Statistics fixtures

`stats_fixtures.py` (called by `generate.py --families stats`) writes one file per family under
`fixtures/stats/`. It needs only these public packages, at these versions: numpy 2.5.3, pandas
3.1.0, scipy 1.18.1, statsmodels 0.15.0, arch 8.0.0 (no TA-Lib, no raptorbt, no path argument).

    uv run --no-project --python 3.12 --with numpy==2.5.3 --with pandas==3.1.0 \
        --with scipy==1.18.1 --with statsmodels==0.15.0 --with arch==8.0.0 \
        python tests/oracle/generate.py --families stats [--only FAMILY ...]

then cut the plain-JSON extracts the Rust tests read (`python3 tests/oracle/extract_stats.py`,
`extract_stats_hmm.py`, `extract_stats_garch.py`, `extract_stats_stationarity_coint.py`, each with
numpy).

Each case records the arguments (`input`), the answer of the reference function in
`stats_reference.py` at full precision (`reference_full`, what the engine must return) and as
plain JSON (`reference`), the library's own numbers (`lib`) and, where one exists, an answer
computed without any library (`known`). A dict of series keeps its order in `input`
(`{"order": [...], "columns": {...}}`), because the JSON writer sorts keys and the first series of
a cointegration is the dependent one.

Where each family's reference comes from (`stats_reference.py` cites the paper at each function):

| family | reference |
|---|---|
| stationarity | statsmodels `adfuller` (constant, AIC lags) and `kpss` (level, automatic lags) |
| coint | statsmodels `OLS`, `coint` and `coint_johansen`; the spread's half-life and the Reinsel-Ahn scaling written from their definitions; a panel whose Johansen eigenvalues fall outside [0, 1) is refused |
| garch | arch `arch_model(...).fit()` on returns in percent; persistence, half-life and the forecast path from the fitted parameters |
| hmm | statsmodels `MarkovRegression(trend="c", switching_variance=True)`; regime order, shares and timeline from its smoothed probabilities |
| hac | statsmodels `OLS(r, 1).fit(cov_type="HAC")` |
| spa | arch `SPA` (stationary bootstrap, the case's seed) |
| predictive | statsmodels `OLS` and its HAC covariance (in `lib`; the case has no `reference`) |
| hurst | written here: rescaled range with the Anis-Lloyd-Peters expectation, and DFA |
| pca | written here: SVD of the standardised panel (numpy), OLS of the target on the scores |
| pbo | written here: combinatorially symmetric cross-validation |
| events | written here: the market-model event study |
| impact | written here: square-root impact, Corwin-Schultz and Roll spreads |
| surface | written here: the parameter-sensitivity surface |
| quality | written here: the bar audit |
| special | scipy `special` and `stats` |

A refusal's wording and error code, the argument limits and the free-text fields of an answer are
raptorbt's API, not statistics: the reference states the limits and refusals as documented and
produces no free text.

`stats_special.json` is plain JSON (hex floats, no gzip): the Rust unit tests read it directly.
`scripts/gen_stat_tables.py` writes `src/stats/tables.rs` and prints the digest that
`tests/stats_tables.rs` pins.

TA-Lib fixtures: regenerating with numpy 2.5.3 changes only the numpy version string in each
header (the committed ones were made with 2.5.4); with `numpy==2.5.4` the TA-Lib files are
byte-identical.
