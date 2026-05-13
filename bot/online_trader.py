import time
import os
import shutil
import pandas as pd
from datetime import datetime
import bot.config as config
from bot.src.data_loader import CryptoDataLoader
from bot.src.processor import ImageProcessor
from bot.src.environment import CryptoTradingEnv
from bot.src.wallet import LiveWallet
from stable_baselines3 import PPO

def save_signal_log(data):
    log_path = os.path.join(config.DATA_DIR, 'live_history.csv')
    df = pd.DataFrame([data])
    df.to_csv(log_path, mode='a', index=False, header=not os.path.exists(log_path))

def run_online_session():
    m_name = f"ppo_master_{config.TIMEFRAME}"
    p_curr = os.path.join(config.MODEL_DIR, f"{m_name}.zip")
    p_orig = os.path.join(config.MODEL_DIR, f"{m_name}_original.zip")
    
    # Backup de segurança antes de começar o online learning
    if os.path.exists(p_curr) and not os.path.exists(p_orig):
        shutil.copyfile(p_curr, p_orig)

    loader = CryptoDataLoader()
    processor = ImageProcessor()
    wallet = LiveWallet()
    
    # Carrega modelo e injeta Learning Rate do config
    model = PPO.load(p_curr, device="cpu")
    model.learning_rate = config.ONLINE_LEARNING_RATE

    candle_count = 0
    while True:
        try:
            prices = {}
            for symbol in config.SYMBOLS:
                df = loader.fetch_historical_data(symbol)
                if df is None or len(df) < config.WINDOW_SIZE: continue
                
                prices[symbol] = df.iloc[-1]['close']
                win = df.iloc[-config.WINDOW_SIZE:].copy().set_index('timestamp')
                obs = processor.dataframe_to_numpy(win)
                
                # IA Decide
                action, _ = model.predict(obs, deterministic=True)
                report = wallet.execute_logic(symbol, int(action), prices[symbol])
                
                print(f"[LIVE] {symbol:10} | {report}")
                
                save_signal_log({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "symbol": symbol, "timeframe": config.TIMEFRAME,
                    "price": prices[symbol], "action": int(action)
                })

                # --- ONLINE LEARNING DINÂMICO ---
                if candle_count >= config.UPDATE_EVERY_N_CANDLES:
                    print(f"[BRAIN] Adaptando modelo aos dados recentes de {symbol}...")
                    # Janela de dados definida no config
                    recent_df = df.tail(config.LOOKBACK_WINDOW_ONLINE)
                    env = CryptoTradingEnv(recent_df, symbol=symbol)
                    
                    model.set_env(env)
                    model.learn(total_timesteps=config.N_STEPS_ONLINE, reset_num_timesteps=False)
                    model.save(p_curr)
                    env.close()

            wallet.save_wallet_log(prices)
            candle_count = 0 if candle_count >= config.UPDATE_EVERY_N_CANDLES else candle_count + 1
            time.sleep(60) # Verifica o mercado a cada minuto
            
        except KeyboardInterrupt: break
        except Exception as e: print(f"Error: {e}"); time.sleep(10)

if __name__ == "__main__":
    run_online_session()
