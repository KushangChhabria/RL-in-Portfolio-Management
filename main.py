import argparse
import numpy as np
from utils.data_loader import load_data
from agents.ppo_agent import PPOAgent
# from agents.sac_agent import SACAgent
# from agents.td3_agent import TD3Agent

AGENT_MAP = {
    'ppo': PPOAgent
    # 'sac': SACAgent,
    # 'td3': TD3Agent
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, choices=AGENT_MAP.keys(), required=True, help="Choose the RL agent to train")
    parser.add_argument('--timesteps', type=int, default=50_000, help="Total training timesteps")
    parser.add_argument('--logdir', type=str, default="./logs", help="TensorBoard log directory")
    args = parser.parse_args()

    tickers = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ITC.NS']

    # Load historical price data
    data = load_data(tickers)

    # Convert data into numpy array format (rows: time, columns: tickers)
    price_array = np.stack([data[t]['Price'].values for t in tickers], axis=1)

    # Instantiate agent
    AgentClass = AGENT_MAP[args.agent]
    agent = AgentClass(price_array=price_array, window_size=60, log_dir=args.logdir)

    # Train, save model
    agent.train(timesteps=args.timesteps)
    agent.save(f"{args.agent}_portfolio")
    print(f"✅ {args.agent.upper()} model saved as {args.agent}_portfolio.zip")
