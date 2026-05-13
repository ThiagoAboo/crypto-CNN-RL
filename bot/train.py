import os
import time
import bot.config as config
from bot.src.data_loader import CryptoDataLoader
from bot.src.environment import CryptoTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Callback customizado para rastrear o progresso
class ProgressLogCallback(BaseCallback):
    def __init__(self, total_steps, symbol, verbose=0):
        super(ProgressLogCallback, self).__init__(verbose)
        self.total_steps = total_steps
        self.symbol = symbol
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % config.N_STEPS == 0: # Atualiza a cada bloco de steps
            percent = (self.n_calls / self.total_steps) * 100
            elapsed = time.time() - self.start_time
            # Estimativa de tempo restante
            if percent > 0:
                total_est = elapsed / (percent / 100)
                remaining = total_est - elapsed
                print(f"[PROGRESS] {self.symbol}: {percent:.1f}% concluído | "
                      f"Restam aprox: {remaining/60:.1f} min")
        return True

def run_training():
    loader = CryptoDataLoader()
    model = None 
    
    if not os.path.exists('bot/models'):
        os.makedirs('bot/models')

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
                device="auto",
                policy_kwargs=dict(normalize_images=False)
            )
        else:
            model.set_env(env)

        # Configura o rastreador de progresso
        callback = ProgressLogCallback(config.TOTAL_TIMESTEPS, symbol)

        print(f"[START] Iniciando processamento de {config.TOTAL_TIMESTEPS} steps...")
        model.learn(total_timesteps=config.TOTAL_TIMESTEPS, reset_num_timesteps=False, callback=callback)

        # --- TRACE RECONSTRUÍDO DO LUCRO FINAL ---
        # Captura o patrimônio líquido final simulado ao término do treino
        pnl_final = ((env.net_worth - config.INITIAL_BALANCE) / config.INITIAL_BALANCE) * 100
        print(f"\n[FINISH] Treino de {symbol} concluído.")
        print(f"[TRACE] Patrimônio Final: ${env.net_worth:.2f} | PnL Acumulado: {pnl_final:.2f}%")
        print(f"{'-'*50}")
        
        # Salva o modelo mestre com o timeframe no nome
        model_path = f"bot/models/ppo_master_{config.TIMEFRAME}"
        model.save(model_path)
        
        print(f"\n[SUCCESS] Aprendizado de {symbol} consolidado no Cérebro Mestre.")
        env.close()

if __name__ == "__main__":
    print(f"[START] Módulo de Treinamento Especializado: {config.TIMEFRAME}")
    run_training()
