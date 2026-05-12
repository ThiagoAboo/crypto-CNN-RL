import time
import os
import pandas as pd
from datetime import datetime
import bot.config as config
from bot.src.data_loader import CryptoDataLoader
from bot.src.processor import ImageProcessor
from stable_baselines3 import PPO

def save_live_log(data):
    """Salva a decisão da IA em um arquivo CSV para auditoria futura."""
    log_path = 'bot/data/live_history.csv'
    
    # Cria a pasta data se não existir
    if not os.path.exists('bot/data'):
        os.makedirs('bot/data')
        
    df_log = pd.DataFrame([data])
    
    # Se o arquivo já existe, anexa sem o cabeçalho. Se não, cria com cabeçalho.
    file_exists = os.path.isfile(log_path)
    df_log.to_csv(log_path, mode='a', index=False, header=not file_exists)

def live_trading():
    loader = CryptoDataLoader()
    processor = ImageProcessor()
    
    model_name = f"ppo_master_{config.TIMEFRAME}"
    model_path = f"bot/models/{model_name}.zip"

    if not os.path.exists(model_path):
        print(f"[ERROR] Modelo {model_name} não encontrado!")
        return

    model = PPO.load(model_path)
    print(f"\n[LIVE] Monitoramento Iniciado: {config.TIMEFRAME}")
    print(f"[INFO] Registrando logs em: bot/data/live_history.csv\n")

    while True:
        try:
            for symbol in config.SYMBOLS:
                df = loader.fetch_historical_data(symbol)
                if df is None or len(df) < config.WINDOW_SIZE: continue
                
                # Prepara imagem
                window = df.iloc[-config.WINDOW_SIZE:].copy()
                window = window.set_index('timestamp')
                obs = processor.dataframe_to_numpy(window)
                
                # IA Decide
                action, _ = model.predict(obs, deterministic=True)
                current_price = df.iloc[-1]['close']
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Tradução amigável
                actions_map = {0: "HOLD/WAIT", 1: "BUY_SIGNAL", 2: "SELL_SIGNAL"}
                decision = actions_map.get(action, "UNKNOWN")

                # Exibe no terminal
                print(f"[{timestamp}] {symbol:10} | Preço: ${current_price:>10.2f} | Ação: {decision}")

                # Salva no Log para análise posterior
                save_live_log({
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "timeframe": config.TIMEFRAME,
                    "price": current_price,
                    "action": decision
                })

            # Intervalo de verificação
            time.sleep(60)

        except KeyboardInterrupt:
            print("\n[STOP] Encerrando monitoramento.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(10)

if __name__ == "__main__":
    live_trading()
