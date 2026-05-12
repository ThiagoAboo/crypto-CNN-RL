import ccxt
import pandas as pd
import bot.config as config

class CryptoDataLoader:
    def __init__(self):
        self.exchange = ccxt.binance()
        if config.DEBUG_TRACE: print("[TRACE] DataLoader inicializado (Binance).")

    def fetch_historical_data(self, symbol):
        if config.DEBUG_TRACE: print(f"[TRACE] Baixando dados de {symbol}...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, config.TIMEFRAME, limit=config.HISTORY_LIMIT)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            if config.DEBUG_LOAD: print(f"[LOAD] SUCCESS - {len(df)} candles carregados para {symbol}.")
            return df
        except Exception as e:
            if config.DEBUG_LOAD: print(f"[LOAD] ERROR - Falha ao baixar {symbol}: {e}")
            return None
