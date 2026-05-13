import os
import time
import bot.config as config
from bot.src.data_loader import CryptoDataLoader
from bot.src.environment import CryptoTradingEnv
from stable_baselines3 import PPO

def run_backtest():
    loader = CryptoDataLoader()
    model_name = f"ppo_master_{config.TIMEFRAME}"
    model_path = f"bot/models/{model_name}.zip"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Modelo {model_name} não encontrado! Treine primeiro.")
        return

    print(f"\n{'='*50}")
    print(f" INICIANDO VALIDATAÇÃO: {config.TIMEFRAME} ")
    print(f"{'='*50}")

    for symbol in config.SYMBOLS:
        df = loader.fetch_historical_data(symbol)
        if df is None: continue

        env = CryptoTradingEnv(df, symbol=symbol)
        model = PPO.load(model_path, env=env)

        obs, _ = env.reset()
        done = False
        
        # Variáveis para controle de progresso
        total_steps = len(df) - config.WINDOW_SIZE
        current_step = 0
        start_time = time.time()

        print(f"[PROCESS] Testando {symbol}...")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            current_step += 1
            
            # Atualiza o progresso a cada 100 passos para não travar o console
            if current_step % 100 == 0 or done:
                percent = (current_step / total_steps) * 100
                print(f"  > {symbol}: {percent:.1f}% concluído...", end='\r')
                
        # --- TRACE RECONSTRUÍDO DO LUCRO NO BACKTEST ---
        # Captura os dados antes de fechar o ambiente de simulação
        pnl_final = ((env.net_worth - config.INITIAL_BALANCE) / config.INITIAL_BALANCE) * 100
        print(f"\n[FINISH] Teste de {symbol} finalizado.")
        print(f"[TRACE] Patrimônio Final: ${env.net_worth:.2f} | PnL do Backtest: {pnl_final:.2f}%")
        print(f"{'-'*50}")

        print(f"\n[SUCCESS] Teste de {symbol} finalizado.")
        env.close()

    print(f"\n{'='*50}")
    print(" TODOS OS TESTES CONCLUÍDOS ")
    print(f"{'='*50}")

if __name__ == "__main__":
    # Mantemos True para ver o [FINISH] com o lucro final de cada moeda
    config.DEBUG = True 
    run_backtest()
