"""``python -m raptorbt``: the analysis commands (see ``raptorbt.analysis.cli``).

``eval``, ``functions``, ``describe``, ``show``, ``export``, ``stats`` and ``skills``; every
command prints JSON and a refusal exits 2 with the error's ``to_dict()``.
"""

from raptorbt.analysis.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
