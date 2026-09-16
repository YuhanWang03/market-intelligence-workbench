"""Unified evaluation bench for Agent V2 and Agent V3.

One question set with one rubric per question, run through both agents under
identical conditions (same model, same tool budget, optionally the same
frozen tool responses), graded by the same judge, and compared pairwise
with the judge blind to which agent wrote which answer.

    python -m v2.agent_bench list
    python -m v2.agent_bench run --mode record --label r1        # live tools, fills the frozen bank
    python -m v2.agent_bench run --mode frozen --label f1 --repeat 3
    python -m v2.agent_bench pair --label f1
    python -m v2.agent_bench report --label f1

Nothing here is a quality score until a run has been reviewed; see README.md.
"""
