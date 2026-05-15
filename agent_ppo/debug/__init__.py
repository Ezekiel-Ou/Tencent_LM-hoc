"""
agent_ppo debug package.

Debug-only utilities used to instrument the env with a fixed-action
script and dump observation statistics. Not used in the training path.
Activated via [debug] section in agent_ppo/conf/train_env_conf.toml.
"""

from agent_ppo.debug.debug_agent import DebugAgent
from agent_ppo.debug.dump_collector import DumpCollector

__all__ = ["DebugAgent", "DumpCollector"]
