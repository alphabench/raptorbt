---
name: pyo3-interop
description: >
  The Rust/Python boundary in raptorbt — PyO3 0.20 bindings, maturin builds,
  the extension-module link trap, releasing the GIL around Rayon, keeping
  _raptorbt.pyi in step with the bindings, and shipping a version bump so the
  consumers actually install it. Use when editing src/python/**, adding or
  changing a #[pyfunction] / #[pyclass], touching Cargo.toml features or
  profiles, changing anything under python/raptorbt/, cutting a release, or
  debugging an ImportError / AttributeError / linker failure that only appears
  from Python.
license: MIT
---

# The Rust/Python boundary

**In plain words: the Rust code and the Python code are compiled separately and
meet at a thin layer of glue. Almost every bug that reaches a user lives in that
glue, not in the trading maths.** The engine can be perfectly correct and still
break the platform, because the wheel that got installed was the wrong one, or a
function's signature changed and the Python caller was never told.

This skill is about that seam. General Rust idiom is the `rust-skills` skill next
door; this one is only the things that are true *because* there is a Python
process on the other side.

## What this crate actually is

| Fact | Value | Why it matters here |
| --- | --- | --- |
| PyO3 | **0.20** | Pre-`Bound<'py, T>`. Most PyO3 answers you will recall or find online target 0.22+ and **will not compile here.** See "PyO3 0.20" below. |
| Edition | **2021**, not 2024 | The `rust-skills` pack targets 1.96 / edition 2024. Its ~17 edition-2024 rules do not apply. |
| Build backend | maturin | `crate-type = ["cdylib", "rlib"]` |
| Consumer | `backend/` as a **published PyPI wheel** | Not a path dependency in production. This is the single most consequential fact in this file. |
| `unsafe` | **zero occurrences** | Keep it that way. If a change seems to need `unsafe`, that is a design discussion, not an implementation detail. |

## The four traps, in order of how much they have actually cost

### 1. The version floor is a floor, so a stale one fails *silently*

This is not hypothetical, and it is the most expensive failure this repo has
produced.

On 2026-08-06 the Docker requirements floors sat at `raptorbt>=0.6.0` while
`pyproject.toml` required `>=0.6.2`. The paper image had a cached 0.6.1 wheel,
which predates position adoption (`initial_positions`). A floor is satisfied by
*any* version at or above it, so the resolver happily kept serving the old
cached wheel and nothing warned. At the morning open all 20 paper deployments
went straight to `Error`.

**What the user saw: every holding on the Portfolio page read "Not under a
strategy" for the whole morning, and nothing traded.**

So, when the engine gains a capability a consuming application depends on:

- Bumping `version` in `Cargo.toml` and `pyproject.toml` here is **step one of
  five, not the whole job.** The other four files
  (`docker/requirements.{paper,trader}.{txt,lock}`) live in the consuming
  application's own repo and are a separate commit there. Read that repo's own
  guidance on bumping the raptorbt version before claiming a release is done.
- **A floor may never point at an unpublished version.** Every image build
  installs from PyPI, so a floor ahead of what is published breaks all of them.
  Publish first, raise floors second.
- **Verify the installed version; never assume it.** A cached layer will satisfy
  a floor with the old wheel:
  ```
  docker exec alphabench__paper_dev python -c "import raptorbt; print(raptorbt.__version__)"
  ```
- Prefer making the new capability **detectable** rather than trusting the
  version string. The probe in the backend's `code_strategy_evaluator.py` refuses
  to run a broker-seeded strategy on an engine that would ignore the seed — that
  refusal is why the 0.6.1 incident was a visible error instead of silently
  wrong P&L. When you add a capability the caller must not run without, give the
  caller a way to ask.

### 2. `extension-module` makes `cargo test` fail to link

`default = ["extension-module"]` is on so that `maturin build` produces an
importable module. But that feature leaves Python symbols to be resolved by the
host interpreter at import time, and a test binary has no interpreter to resolve
them against — so linking fails outright.

**Every Rust test and clippy invocation must pass `--no-default-features`:**

```bash
cargo test  --no-default-features --all-targets
cargo clippy --no-default-features --all-targets
cargo fmt --check
```

CI already does this (`.github/workflows/ci.yml`). If you see an inscrutable
linker error mentioning `_Py...` symbols, this is the cause, not your code. The
comment in `Cargo.toml` records it; do not "fix" it by removing the feature.

### 3. A panic crossing the FFI boundary is not a Python exception

Rust errors reach Python through exactly one conversion, in `src/core/error.rs`:

```rust
impl From<RaptorError> for pyo3::PyErr {
    fn from(err: RaptorError) -> pyo3::PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
```

That is the supported path: return `PyResult<T>`, let `?` do the work, and the
caller gets a `ValueError` it can catch. A **panic** is not that. It unwinds into
PyO3, which converts it to `PanicException` — which is not a `ValueError`, is not
caught by the backend's handlers, and carries a message written for a Rust
developer rather than a trader.

There are 223 `unwrap`/`expect`/`panic!`/`unreachable!` sites in `src/`, but only
6 in `src/python/`. **That ratio is the invariant worth protecting.** Deep in the
maths, an `expect` on a genuine invariant is defensible. In `src/python/**`,
reached directly from user input, it is a crash with a bad error message.

- New code in `src/python/**` returns `PyResult`; it does not `unwrap`.
- An `expect` there needs a comment proving the case is unreachable *given
  validation already performed in that same function*.
- Financial edge cases are inputs, not invariants: empty bar series, a single
  bar, all-NaN columns, zero capital, zero-width spreads. There are 176 NaN
  sites in `src/` — NaN is a value this engine genuinely handles, so a NaN
  arriving from Python is never grounds to panic.

### 4. The `.pyi` stub is a promise, and nothing checks it

`python/raptorbt/_raptorbt.pyi` is 795 lines of hand-maintained type stubs, and
`py.typed` tells consumers to trust them. **No test compares the stub to the
actual `#[pyfunction]` signatures.** A renamed parameter or a changed return type
in Rust leaves the stub lying, and the backend's type checker keeps agreeing with
the lie until something fails at runtime.

So a binding change is at minimum three edits in one commit:

1. the `#[pyfunction]` / `#[pyclass]` in `src/python/**`
2. the matching entry in `_raptorbt.pyi`
3. a behavioural test in `tests/python/` that calls it *through the wheel*

`cargo test` cannot catch a stub drift, and it cannot catch a signature change
either — only `tests/python/` runs the real boundary. That is what the
`build` CI job (`.github/workflows/ci.yml`), which installs the wheel and runs `pytest tests/python`, is for.

## PyO3 0.20 — what will not compile

Pinned to `pyo3 = "0.20"`, released before the `Bound<'py, T>` API landed in 0.21
and became the norm in 0.22+. Recalled or web-sourced examples will mostly be
newer. In this crate:

- Gil-bound references are plain `&`: `#[pymodule] fn _raptorbt(_py: Python<'_>,
  m: &PyModule)` in `src/lib.rs` is the shape. In 0.22+ that same signature is
  `&Bound<'_, PyModule>`. There are **zero** `Bound<` occurrences in this crate;
  if you introduce one, it will not compile.
- Likewise `&PyList` / `&PyDict`, not `&Bound<'_, PyList>`. The crate touches
  these rarely — mostly it moves `PyObject` and numpy arrays — so there is
  little in-tree precedent to copy from. Check the PyO3 **0.20** docs
  specifically, not the current ones.
- `Python::with_gil(|py| ...)` and `py.allow_threads(...)` — stable across these
  versions, safe to copy from anywhere.
- `#[pyo3(get, set)]`, `#[new]`, `#[staticmethod]`, `#[pyfunction]` — unchanged.

If a snippet uses `Bound`, `.as_borrowed()`, or `IntoPyObject`, it is written for
a newer PyO3. Translate it rather than upgrading the dependency in passing — a
PyO3 major bump is its own change, with its own release, and it touches every
file in `src/python/`.

## Releasing the GIL around Rayon

Three sites call `py.allow_threads` (`src/python/bindings.rs` ×2,
`src/python/portfolio_bindings.rs` ×1), each immediately around an
`into_par_iter()`. That pairing is the rule:

**Rayon parallelism inside a binding is only real if the GIL is released first.**
Without `allow_threads`, the worker threads serialise behind the interpreter lock
and the parallelism is a no-op that still pays the scheduling cost.

The constraint that comes with it: **nothing inside the `allow_threads` closure
may touch Python.** No `PyAny`, no `PyErr`, no `py` token. Convert Python inputs
into owned Rust values *before* the closure and build the Python result *after*
it — which is exactly why the existing sites collect into
`Vec<(String, Result<...>)>` and convert afterwards. Follow that shape; a
`PyErr` constructed inside the closure is a deadlock waiting to happen.

## Numeric behaviour is a contract, not an implementation detail

This engine reports money. Two rules follow, and they outrank tidiness:

- **NaN means "absent", never zero.** The backend states this as a
  non-negotiable for Greeks, and the same holds through the engine. Silently
  coercing NaN to 0.0 turns missing data into a real position worth nothing,
  which is a wrong number presented as a fact.
- **A sign error is a total loss of trust.** `git log` here records a spread
  backtest that reported P&L with the wrong sign. When touching P&L, drawdown,
  or optimizer output, add the golden test *first* — `tests/python/golden/`
  exists for exactly this, and a golden file is cheaper than a user discovering
  it.

## Before you call an engine change done

```bash
cargo fmt --check
cargo clippy --no-default-features --all-targets
cargo test  --no-default-features --all-targets
maturin develop                 # or: make dev-up-engine, from backend/
uv run pytest tests/python -v   # the only layer that exercises the real boundary
```

Then ask the four questions the Rust tests cannot answer:

1. Did the `.pyi` stub change alongside the signature?
2. Is there a `tests/python/` test that calls this through the wheel?
3. If the backend needs this capability, is the version published, are all four
   requirements files raised, and can the caller *detect* the capability rather
   than trusting a version string?
4. Can this new code path panic on user input instead of returning `PyResult`?

## What a user sees

Every change here reaches someone eventually. State it plainly when reporting:
"backtests including options now report the correct P&L sign" is a consequence;
"updated the spread strategy module" is not. And a change that is built but not
published, or published but not floored in the backend's requirements files, is
**not shipped** — it is invisible to every user until those commits land too.
