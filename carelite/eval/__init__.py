"""Evaluation subsystem for CARELite AI.

Deliberately empty of re-exports. Several lanes add subpackages here in
parallel (`rubric`, `judge`, `human`, `smoke`), and an aggregating
``__init__`` would be a permanent merge conflict between them. Import from the
subpackage you need:

    from carelite.eval.rubric import DIMENSIONS, score_text
"""
