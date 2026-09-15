"""
Package marker for importing modules under `scripts/`.

This repository historically treated `scripts/` as a plain directory and
manipulated `sys.path` inside scripts. Some experiments benefit from imports like
`from scripts.truthfulqa_paper_audit import ...`, so we provide this minimal
`__init__.py` to support that style without changing existing behavior.
"""

