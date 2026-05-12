import os
import bot.config as config
from bot.src.data_loader import CryptoDataLoader
from bot.src.environment import CryptoTradingEnv
from stable_baselines3 import PPO

def evaluate(path, env):
    model = PPO.load(path, env=env)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.net_worth

def run_compare():
    loader = CryptoDataLoader()
    p_orig = f"bot/models/ppo_master_{config.TIMEFRAME}_original.zip"
    p_evol = f"bot/models/ppo_master_{config.TIMEFRAME}.zip"

    if not os.path.exists(p_orig): 
        print("Cópia original não encontrada."); return

    for symbol in config.SYMBOLS:
        df = loader.fetch_historical_data(symbol)
        if df is None: continue
        
        env = CryptoTradingEnv(df, symbol=symbol)
        v_orig = evaluate(p_orig, env)
        v_evol = evaluate(p_evol, env)
        
        diff = ((v_evol - v_orig) / v_orig) * 100
        print(f"\n[{symbol}] Original: ${v_orig:.2f} | Evoluído: ${v_evol:.2f}")
        print(f"Resultado: {'MELHOROU' if diff > 0 else 'PIOROU'} {abs(diff):.2f}%")

if __name__ == "__main__":
    run_compare()
