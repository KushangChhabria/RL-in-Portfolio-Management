import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from env.portfolio_env import PortfolioEnv
from agents.ppo_agent import PPOAgent
from utils.metrics import calculate_sharpe, calculate_volatility, calculate_cumulative_return

def plan_strategy(price_array, target_return, time_horizon_days, risk_level,
                  window_size=30, timesteps=50_000, strategy_mode="long_short"):
    """
    Simulate PPO-based strategy and check feasibility for given user goal.
    """
    risk_penalty_map = {
        "low": (0.2, 0.1),
        "medium": (0.1, 0.05),
        "high": (0.05, 0.01)
    }
    dd_penalty, vol_penalty = risk_penalty_map.get(risk_level.lower(), (0.1, 0.05))

    # Ensure enough data
    min_required = time_horizon_days + window_size + 1
    if len(price_array) < min_required:
        return {
            "success": False,
            "message": f"Not enough data: need at least {min_required} days, got {len(price_array)}"
        }

    planning_data = price_array[-(time_horizon_days + window_size):]

    agent = PPOAgent(price_array=planning_data, window_size=window_size)
    agent.env.max_drawdown_penalty = dd_penalty
    agent.env.volatility_penalty = vol_penalty
    agent.env.strategy_mode = strategy_mode
    agent.train(timesteps=timesteps)

    env = PortfolioEnv(price_array=planning_data, window_size=window_size,
                   max_drawdown_penalty=dd_penalty,
                   volatility_penalty=vol_penalty,
                   strategy_mode=strategy_mode) 

    obs = env.reset()
    done = False
    portfolio_values = []

    model = agent.model

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        portfolio_values.append(info['portfolio_value'])

    values = np.array(portfolio_values)
    actual_return = calculate_cumulative_return(values)
    sharpe = calculate_sharpe(np.diff(values) / values[:-1])
    volatility = calculate_volatility(np.diff(values) / values[:-1])

    if actual_return >= target_return:
        return {
            "success": True,
            "message": f"Goal achievable. Projected return: {actual_return*100:.2f}%",
            "return": actual_return,
            "sharpe": sharpe,
            "volatility": volatility,
            "recommended_weights": env.weights.tolist()
        }
    else:
        return {
            "success": False,
            "message": f"Target return of {target_return*100:.1f}% not feasible. Expected: {actual_return*100:.2f}%",
            "return": actual_return,
            "sharpe": sharpe,
            "volatility": volatility
        }

def replan_strategy(previous_plan, rejection_reason=None):
    print(" Replanning strategy...")
    original_risk = previous_plan.get("risk_level", "medium")
    new_risk = "low" if original_risk == "medium" else "very_low" if original_risk == "low" else "medium"
    new_target = previous_plan["requested_return"] * 0.8
    new_duration = previous_plan["duration_days"]
    tickers = previous_plan["tickers"]

    new_plan = plan_strategy(
        tickers=tickers,
        target_return=new_target,
        duration_days=new_duration,
        risk_level=new_risk
    )

    new_plan["replanned_from"] = previous_plan
    new_plan["rejection_reason"] = rejection_reason
    return new_plan

def suggest_strategy(goal_profit: float, time_horizon: int, risk_tolerance: str, strategy_type: str = "long_short"):
    dummy_prices = np.random.rand(200 + time_horizon, 5) * 100  # [days x assets]
    result = plan_strategy(
        price_array=dummy_prices,
        target_return=goal_profit,
        time_horizon_days=time_horizon,
        risk_level=risk_tolerance,
        strategy_mode=strategy_type 
    )
    result["risk_level"] = risk_tolerance
    result["requested_return"] = goal_profit
    result["duration_days"] = time_horizon
    result["strategy_type"] = strategy_type
    result["tickers"] = ["DUMMY1", "DUMMY2", "DUMMY3", "DUMMY4", "DUMMY5"]
    return result
