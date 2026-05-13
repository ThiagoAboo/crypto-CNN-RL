import ccxt
import pandas as pd
import bot.config as config

class CryptoDataLoader:
    def __init__(self):
        # Localmente usamos a Binance porque não há bloqueio de IP no Brasil
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        if config.DEBUG_TRACE: print("[TRACE] DataLoader local inicializado (Binance).")

    def fetch_historical_data(self, symbol):
        if config.DEBUG_TRACE: print(f"[TRACE] Baixando dados locais de {symbol}...")
        try:
            # Puxa os 5000 candles diretos em uma única chamada rápida
            ohlcv = self.exchange.fetch_ohlcv(symbol, config.TIMEFRAME, limit=config.HISTORY_LIMIT)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            if config.DEBUG_LOAD: 
                print(f"[LOAD] SUCCESS - {len(df)} candles carregados para {symbol} via Binance local.")
            return df
        except Exception as e:
            if config.DEBUG_LOAD: print(f"[LOAD] ERROR - Falha local ao baixar {symbol} via Binance: {e}")
            return None
