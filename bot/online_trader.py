import sys
import os
import numpy as np

# --- ANCORAGEM DE DIRETÓRIO LOCAL ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# PONTE DE COMPATIBILIDADE KAGGLE -> LOCAL
import bot.src.models as local_models
sys.modules['bot.src.cnn'] = local_models

import time
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
    """Salva o histórico de sinais APENAS se for uma execução real (BUY ou SELL)."""
    log_path = os.path.join(config.DATA_DIR, 'live_history.csv')
    df = pd.DataFrame([data])
    df.to_csv(log_path, mode='a', index=False, header=not os.path.exists(log_path))

def run_online_session():
    m_name = f"ppo_master_{config.TIMEFRAME}"
    p_curr = os.path.join(config.MODEL_DIR, f"{m_name}.zip")
    p_orig = os.path.join(config.MODEL_DIR, f"{m_name}_original.zip")
    
    if os.path.exists(p_curr) and not os.path.exists(p_orig):
        shutil.copyfile(p_curr, p_orig)

    loader = CryptoDataLoader()
    processor = ImageProcessor()
    wallet = LiveWallet()
    
    model = PPO.load(p_curr, device="cpu")
    model.learning_rate = config.ONLINE_LEARNING_RATE

    print(f"\n[LIVE] Monitoramento Iniciado: {config.TIMEFRAME} via Binance Local")
    print(f"[INFO] Filtro por Retorno de Execução Ativado.\n")

    candle_count = 0
    while True:
        try:
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            prices = {}
            for symbol in config.SYMBOLS:
                df = loader.fetch_historical_data(symbol)
                if df is None or len(df) < config.WINDOW_SIZE: continue
                
                prices[symbol] = df.iloc[-1]['close']
                win = df.iloc[-config.WINDOW_SIZE:].copy().set_index('timestamp')
                obs = processor.dataframe_to_numpy(win)
                
                if obs.ndim == 3:
                    obs = np.expand_dims(obs, axis=0)
                
                action, _ = model.predict(obs, deterministic=True)
                
                if hasattr(action, "item"):
                    trade_action = int(action.item())
                else:
                    trade_action = int(action)
                
                # Executa a lógica financeira e captura o relatório em texto
                report = wallet.execute_logic(symbol, trade_action, prices[symbol])
                print(f"[{current_timestamp.split()[-1]}] {symbol:10} | {report}")
                
                # --- NOVO FILTRO PRECISO POR EXTENSO ---
                # Ignora o número bruto (0,1,2) e valida o que a carteira REALMENTE executou
                if "BUY" in report.upper():
                    save_signal_log({
                        "timestamp": current_timestamp,
                        "symbol": symbol, "timeframe": config.TIMEFRAME,
                        "price": prices[symbol], "action": "BUY" # Grava o texto literal
                    })
                elif "SELL" in report.upper():
                    save_signal_log({
                        "timestamp": current_timestamp,
                        "symbol": symbol, "timeframe": config.TIMEFRAME,
                        "price": prices[symbol], "action": "SELL" # Grava o texto literal
                    })

                if candle_count >= config.UPDATE_EVERY_N_CANDLES:
                    recent_df = df.tail(config.LOOKBACK_WINDOW_ONLINE)
                    env = CryptoTradingEnv(recent_df, symbol=symbol)
                    model.set_env(env)
                    model.learn(total_timesteps=config.N_STEPS_ONLINE, reset_num_timesteps=False)
                    model.save(p_curr)
                    env.close()

            log_path = os.path.join(config.DATA_DIR, 'wallet_history.csv')
            wallet_data = {
                "timestamp": current_timestamp,
                "balance": wallet.balance,
                "net_worth": wallet.get_total_net_worth(prices)
            }
            df_wallet_log = pd.DataFrame([wallet_data])
            df_wallet_log.to_csv(log_path, mode='a', index=False, header=not os.path.exists(log_path))

            candle_count = 0 if candle_count >= config.UPDATE_EVERY_N_CANDLES else candle_count + 1
            time.sleep(60) 
            
        except KeyboardInterrupt: 
            print("\n[STOP] Encerrando monitoramento.")
            break
        except Exception as e: 
            print(f"Error no Loop Principal: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_online_session()
