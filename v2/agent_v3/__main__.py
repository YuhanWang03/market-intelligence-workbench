"""Run `python -m v2.agent_v3 --demo` without keys or network."""

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="LangGraph Agent V3")
    parser.add_argument("text", nargs="?")
    parser.add_argument("--demo", action="store_true", help="run the fixed offline NVDA fixture")
    parser.add_argument("--graph", action="store_true", help="print the actual compiled graph as Mermaid")
    parser.add_argument("--session", default="cli")
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--enable-mutations", action="store_true", help="register existing watchlist/alert writes; still requires confirmation")
    parser.add_argument("--confirm", metavar="RUN_ID")
    parser.add_argument("--reject", metavar="RUN_ID")
    parser.add_argument("--data-dir")
    parser.add_argument("--env-file", help="explicitly load an environment file for live mode")
    parser.add_argument("--max-seconds", type=float, default=180)
    args = parser.parse_args(argv)
    if args.confirm and args.reject:
        parser.error("choose --confirm or --reject")
    if (args.confirm or args.reject) and (args.text or args.demo or args.graph):
        parser.error("confirmation cannot be combined with a new question, demo or graph output")
    if args.demo or args.graph:
        from v2.agent_v3.demo import DEMO_QUESTION, build_demo_agent
        agent = build_demo_agent()
        if args.graph:
            print(agent.graph.get_graph().draw_mermaid())
            return 0
        if args.text:
            parser.error("--demo uses a fixed fixture; omit text")
        result = agent.run(DEMO_QUESTION, session_id=args.session)
    else:
        if not (args.text or args.confirm or args.reject):
            parser.error("provide a question, --demo, --graph or a confirmation run ID")
        if args.env_file:
            from dotenv import load_dotenv
            load_dotenv(args.env_file, override=False)
        from v2.agent_v3.graph import AgentV3Config
        from v2.agent_v3.runtime import build_workspace_agent
        agent = build_workspace_agent(config=AgentV3Config(enable_web=args.web, max_seconds=args.max_seconds), data_dir=args.data_dir, enable_mutations=args.enable_mutations)
        if args.confirm or args.reject:
            result = agent.resume(session_id=args.session, run_id=args.confirm or args.reject, approve=bool(args.confirm))
        else:
            result = agent.run(args.text, session_id=args.session, allow_web=args.web)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    agent.store.close()
    if hasattr(agent, "checkpoint_connection"):
        agent.checkpoint_connection.close()
    return 1 if result.status.value == "failed" else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
