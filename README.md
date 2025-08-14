# 📈 Reinforcement Learning for Portfolio Management

A professional research-driven system that applies **Deep Reinforcement Learning (DRL)** to optimize multi-asset portfolios.  
The platform supports **PPO**, **TD3**, and **SAC** agents, multiple strategy modes, goal-based investing, and live market data simulation.  
Developed as part of my **Summer Research Internship**, this project bridges academic research and real-world financial trading systems.

---

## 🚀 Key Features

- **Multiple DRL Agents**
  - PPO, TD3, SAC implemented using a unified `BaseAgent` interface.
- **Custom Portfolio Environment**
  - Long-only, long-short, and hedged strategies.
  - Reward shaping to balance profitability, volatility, and drawdown.
  - Log-return based state representation for numerical stability.
- **Goal-Based Investing**
  - Set profit target, duration, and risk level.
  - Automatic strategy replanning when goals are unachievable.
- **Data Integration**
  - Historical market data for training.
  - Real-time-like data streaming via `yfinance` (modular for easy broker API integration like Zerodha Kite or Alpaca).
- **Advanced Evaluation**
  - Sharpe ratio, volatility, max drawdown, and annualized returns.
  - Visualizations: portfolio weight heatmaps, trade frequency analysis, and equity curves.
- **Interactive Dashboard**
  - Streamlit-based interface for strategy simulation.
  - Compare agent strategies against baseline benchmarks.

---

## 📂 Project Structure

.
├── agents/ # PPO, TD3, SAC agent implementations
├── env/ # Custom OpenAI Gym-like portfolio environment
├── train/ # Standalone training scripts
├── utils/ # Data handling, metrics, plotting, broker APIs, etc.
├── scripts/ # Testing and visualization utilities
├── logs/ # Training and evaluation logs
├── main.py # CLI entry point for training and evaluation
├── evaluate.py # Evaluate trained agents
├── optuna_tune.py # Hyperparameter tuning via Optuna
├── streamlit_app.py # Streamlit dashboard for strategy simulation
└── requirements.txt # Python dependencies
---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rl-portfolio-management.git
cd rl-portfolio-management

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```
---

## 📊 Training Agents
```bash
#Train a PPO agent:
python train/train_ppo.py

#Train a TD3 agent:
python train/train_td3.py

#Train a SAC agent:
python train/train_sac.py
```
---

##🖥 Interactive Dashboard
```bash
#Launch the Streamlit dashboard:
streamlit run streamlit_app.py
```
---
##📈 Evaluation
```bas
#Evaluate a trained agent on test data:
python evaluate.py --agent PPO --model_path ppo_portfolio.zip
```
---
##📜 Research Context

-**This system was developed to:**
  - Explore DRL’s ability to adapt to dynamic market conditions.
  - Integrate financial risk measures directly into the reward function.
  - Build a modular, production-ready RL trading system with real-time capabilities.
  - It combines quantitative finance techniques with deep reinforcement learning algorithms to move beyond static optimization and towards adaptive decision-making.
  
  ---
##🛠 Tech Stack

-Python 3.9+
-Stable-Baselines3
-Pandas / NumPy
-Matplotlib
-Optuna
-Streamlit
-yfinance
---

##📚 References

-Jiang, M., Xu, D., Liang, Y., “Deep Reinforcement Learning for Trading,” 2017.
-OpenAI Spinning Up Documentation.
-Stable-Baselines3 Documentation.
---
