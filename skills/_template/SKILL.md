---
name: raptorbt-<domain>
description: >
    One or two sentences: what an agent can do with this domain.
version: 1.0
api_version: 2.x
depends_on:
    - raptorbt-core
status: draft
---

# RaptorBT <Domain>

## 1. When to use this skill

The situations where this API domain is needed, and the ones where it is not.

## 2. Domain overview

The concepts and the supported functionality, in the order an agent needs them.

## 3. API reference

Public classes, functions, arguments and return values. Public names only
(everything importable from `raptorbt` or a documented submodule).

## 4. Standard workflow

The correct sequence of operations, numbered.

## 5. Examples

Executable, tested Python. Every `python` block runs in a subprocess under
`python -m raptorbt.skills check`. An example of invalid usage must say so
with the marker `<!-- skill: invalid -->` on the line above its fence, and it
must fail when run.

## 6. Composition

How this domain interacts with the others.

## 7. Constraints and correctness

Ordering, validation, event-time semantics and numerical assumptions.

## 8. Common mistakes

Invalid or dangerous usage, each with the error it raises or the wrong result
it gives.

## 9. Validation

How an agent tests its implementation before trusting it, and what it must
verify before live deployment.

## 10. Related skills

Links to other skills, as relative paths: `[Orders](../orders/SKILL.md)`.
