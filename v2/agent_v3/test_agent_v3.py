"""Original V2 question corpus comparison entry for Agent V3.

First capture: python -m v2.agent_eval_capture --output data/agent_v3/corpus.json
Then compare: python -m v2.agent_v3.test_agent_v3 --corpus data/agent_v3/corpus.json
              --output-dir data/agent_v3/original-comparison

Default: offline original-plan/fixture replay. Add --live-model for native
model planning and answers with frozen business tools.
The external driver invokes both peers; Agent V3 never invokes Agent V2.
"""
from v2.agent_original_comparison import main


if __name__ == '__main__':
    main()
