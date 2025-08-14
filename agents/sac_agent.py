from stable_baselines3 import SAC
from agents.base_agent import BaseAgent

class SACAgent(BaseAgent):
    def __init__(self, price_array, window_size=60, log_dir=None,
                 max_drawdown_penalty=0.1, volatility_penalty=0.05,
                 use_optuna=False, optuna_params_path="optuna_sac_params.json"):
        super().__init__(
            model_class=SAC,
            price_array=price_array,
            window_size=window_size,
            log_dir=log_dir,
            max_drawdown_penalty=max_drawdown_penalty,
            volatility_penalty=volatility_penalty,
            use_optuna=use_optuna,
            optuna_params_path=optuna_params_path
        )
