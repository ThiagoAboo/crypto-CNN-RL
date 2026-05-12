import pandas as pd
import os
import json
from datetime import datetime
import bot.config as config

class LiveWallet:
    def __init__(self):
        self.state_path = os.path.join(config.DATA_DIR, 'wallet_state.json')
        
        if config.RESET_WALLET or not os.path.exists(self.state_path):
            self._initial_state()
            if config.DEBUG: print("[WALLET] Carteira reiniciada/inicializada.")
        else:
            self._load_state()
            if config.DEBUG: print("[WALLET] Estado anterior carregado com sucesso.")

    def _initial_state(self):
        """Define o estado inicial padrão."""
        self.balance = config.INITIAL_BALANCE
        self.positions = {symbol: 0.0 for symbol in config.SYMBOLS}
        self.entry_prices = {symbol: 0.0 for symbol in config.SYMBOLS}
        self._save_state()

    def _save_state(self):
        """Salva o saldo e posições em um arquivo JSON."""
        state = {
            "balance": self.balance,
            "positions": self.positions,
            "entry_prices": self.entry_prices
        }
        with open(self.state_path, 'w') as f:
            json.dump(state, f)

    def _load_state(self):
        """Lê o arquivo JSON para recuperar a carteira."""
        with open(self.state_path, 'r') as f:
            state = json.load(f)
            self.balance = state["balance"]
            self.positions = state["positions"]
            self.entry_prices = state["entry_prices"]

    def execute_logic(self, symbol, action, price):
        result = "WAITING"
        if action == 1 and self.positions[symbol] == 0:
            result = self._buy(symbol, price)
            self._save_state() # Salva após comprar
        elif action == 2 and self.positions[symbol] > 0:
            result = self._sell(symbol, price)
            self._save_state() # Salva após vender
        
        elif self.positions[symbol] > 0:
            pnl = (price - self.entry_prices[symbol]) / self.entry_prices[symbol]
            result = f"HOLDING ({pnl*100:.2f}%)"
            
        return result

    def _buy(self, symbol, price):
        amount = config.INITIAL_BALANCE * config.ALLOCATION_PER_TRADE
        if self.balance < amount + config.MIN_BALANCE_RESERVE:
            return "LOW BALANCE"

        cost = amount * config.COMMISSION
        self.positions[symbol] = (amount - cost) / price
        self.balance -= amount
        self.entry_prices[symbol] = price
        return f"BUY at ${price:.2f}"

    def _sell(self, symbol, price):
        pnl = (price - self.entry_prices[symbol]) / self.entry_prices[symbol]
        revenue = (self.positions[symbol] * price) * (1 - config.COMMISSION)
        self.balance += revenue
        self.positions[symbol] = 0.0
        self.entry_prices[symbol] = 0.0
        return f"SELL | PnL: {pnl*100:.2f}%"

    def get_total_net_worth(self, current_prices):
        crypto_val = sum(self.positions[s] * current_prices.get(s, 0) for s in config.SYMBOLS)
        return self.balance + crypto_val

    def save_wallet_log(self, current_prices):
        log_path = os.path.join(config.DATA_DIR, 'wallet_history.csv')
        data = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "balance": self.balance,
            "net_worth": self.get_total_net_worth(current_prices)
        }
        df = pd.DataFrame([data])
        df.to_csv(log_path, mode='a', index=False, header=not os.path.exists(log_path))
