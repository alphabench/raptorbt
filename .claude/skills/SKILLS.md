# Skills — raptorbt

Two skills. One vendored, one written for this crate.

| Skill | Origin | Tracked? | What it covers |
| --- | --- | --- | --- |
| `pyo3-interop/` | Written for this repo | **yes** | The Rust/Python seam: PyO3 0.20 API shapes, the `extension-module` link trap, GIL release around Rayon, `.pyi` stub drift, and the version-floor rule that caused the 2026-08-06 paper outage. |
| `rust-skills/` | [leonardomso/rust-skills](https://github.com/leonardomso/rust-skills) v1.5.1, MIT | no — gitignored | 265 general Rust rules across 26 categories. The 499-line `SKILL.md` is an index; the 1.2MB of rules under `rules/` load only when one is followed, so idle context cost is small. |

`rust-skills/` is deliberately **not** in git: 1.2MB of third-party content that a
clone restores in seconds is not worth carrying in this repo's history. If the
directory is missing (fresh clone, `git clean`, new machine), restore it with:

```bash
git clone --depth 1 https://github.com/leonardomso/rust-skills /tmp/rust-skills
mkdir -p .claude/skills/rust-skills
cp /tmp/rust-skills/SKILL.md /tmp/rust-skills/LICENSE .claude/skills/rust-skills/
cp -r /tmp/rust-skills/rules .claude/skills/rust-skills/rules
```

`pyo3-interop/` **is** tracked, and the distinction is the point: it is not an
installed dependency but this repo's own record of what has actually gone wrong
at the binding boundary. Losing it loses the knowledge, not a download.

## Reading `rust-skills` against this crate

It targets **Rust 1.96, edition 2024**. This crate is **edition 2021**. About 17
of the 265 rules assume edition-2024 or 1.85+ features; the other ~248 are
edition-independent and apply directly. Where the two disagree, this crate wins
— do not migrate the edition to satisfy a rule.

Two more places it needs local translation:

- **`unsafe-` rules (7).** This crate has **zero** `unsafe`. They are worth
  reading as a reason to keep it that way, not as an invitation.
- **`async-` rules (18) and `conc-` rules.** There is no async runtime here; no
  tokio, no `async fn`. Parallelism is Rayon under `py.allow_threads`, which the
  `pyo3-interop` skill covers and this pack does not.

Nothing in `rust-skills` mentions PyO3, maturin, abi3, or the GIL — that gap is
exactly why `pyo3-interop` exists.

## What was evaluated and not installed

- **Impertio-Studio/Rust-Claude-Skill-Package** (44 skills) — explicitly
  "Requires Rust 1.85+, edition 2024" in every skill's frontmatter, and ★0. Its
  strongest sections (proc-macros, no-std, clap CLI, tokio, FFI-bindgen) cover
  things this crate does not do.
- **VGVentures/rust-code-assessment** — a one-shot audit producing
  `RUST_ASSESSMENT.md`. Reasonable, but it assumes tokio and workspaces, and
  much of its checklist (clippy, fmt, CI gates) is already enforced in
  `.github/workflows/ci.yml`.
- **photostructure/coding-skills** (rust plugin) — project setup, packaging, and
  publishing readiness. This crate is already set up and published.
- **jeffallan/claude-skills → rust-engineer**, **molaco/rust-code-mcp**,
  **nanlong/rust-architect** — general seniority guidance and an MCP server for
  semantic search; neither adds anything the above two do not.

None of the above mention PyO3 or maturin either. Checked, not assumed.
