import os
import time
import bot.config as config
from bot.src.data_loader import CryptoDataLoader
from bot.src.environment import CryptoTradingEnv
from bot.src.models import CryptoCNN 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class ProgressLogCallback(BaseCallback):
    def __init__(self, total_steps, symbol, verbose=0):
        super(ProgressLogCallback, self).__init__(verbose)
        self.total_steps = total_steps
        self.symbol = symbol
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % config.N_STEPS == 0:
            percent = (self.n_calls / self.total_steps) * 100
            elapsed = time.time() - self.start_time
            if percent > 0:
                total_est = elapsed / (percent / 100)
                remaining = total_est - elapsed
                print(f"[PROGRESS] {self.symbol}: {percent:.1f}% concluído | Restam aprox: {remaining/60:.1f} min")
        return True

def run_training():
    loader = CryptoDataLoader()
    model = None 
    
    if not os.path.exists('bot/models'):
        os.makedirs('bot/models')

    # Dicionário explícito para forçar o PPO a usar a nossa rede leve de 32x32
    policy_kwargs = dict(
        features_extractor_class=CryptoCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False 
    )

    for symbol in config.SYMBOLS_TRANING:
        print(f"\n{'='*50}")
        print(f" FOCO DE TREINO: {symbol} ")
        print(f"{'='*50}")

        df = loader.fetch_historical_data(symbol)
        if df is None: continue

        env = CryptoTradingEnv(df, symbol=symbol)

        if model is None:
            model = PPO(
                "CnnPolicy", 
                env, 
                verbose=1 if config.DEBUG_TRAINING else 0,
                learning_rate=config.LEARNING_RATE,
                n_steps=config.N_STEPS,
                batch_size=config.BATCH_SIZE,
                gamma=config.GAMMA,
                device="cpu", 
                policy_kwargs=policy_kwargs 
            )
        else:
            model.set_env(env)

        callback = ProgressLogCallback(config.TOTAL_TIMESTEPS, symbol)

        print(f"[START] Iniciando processamento de {config.TOTAL_TIMESTEPS} steps...")
        model.learn(total_timesteps=config.TOTAL_TIMESTEPS, reset_num_timesteps=False, callback=callback)

        pnl_final = ((env.net_worth - config.INITIAL_BALANCE) / config.INITIAL_BALANCE) * 100
        print(f"\n[FINISH] Treino de {symbol} concluído.")
        print(f"[TRACE] Patrimônio Final: ${env.net_worth:.2f} | PnL Acumulado: {pnl_final:.2f}%")
        print(f"{'-'*50}")

        model_path = f"bot/models/ppo_master_{config.TIMEFRAME}"
        model.save(model_path)
        env.close()

if __name__ == "__main__":
    config.DEBUG = True 
    print(f"[START] Módulo de Treinamento Especializado: {config.TIMEFRAME}")
    run_training()
