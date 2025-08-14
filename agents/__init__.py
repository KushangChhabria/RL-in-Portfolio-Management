# This file marks the folder as a Python package
from .base_agent import BaseAgent

def get_agent(agent_name, env, log_dir="./logs"):
    return BaseAgent(env, algo=agent_name, log_dir=log_dir)
