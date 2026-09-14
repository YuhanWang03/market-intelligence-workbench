"""Run the zero-network Agent V2 seed evaluation."""

from v2.agent_v2.eval.runner import render, run_suite


def main() -> int:
    report = run_suite()
    print(render(report))
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
