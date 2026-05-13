import gymnasium as gym
from gymnasium import spaces
import numpy as np
import bot.config as config
from .processor import ImageProcessor

class CryptoTradingEnv(gym.Env):
    def __init__(self, df, symbol="Unknown"):
        super(CryptoTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.symbol = symbol
        self.processor = ImageProcessor()
        
        self.action_space = spaces.Discrete(3)
        h, w = config.IMG_SIZE
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, h, w), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = config.WINDOW_SIZE
        self.balance = config.INITIAL_BALANCE
        self.shares_held = 0
        self.net_worth = config.INITIAL_BALANCE
        self.prev_net_worth = config.INITIAL_BALANCE
        self.entry_price = 0
        return self._get_observation(), {}

    def _get_observation(self):
        if self.current_step >= len(self.df):
            h, w = config.IMG_SIZE
            return np.zeros((1, h, w), dtype=np.float32)
        window = self.df.iloc[self.current_step - config.WINDOW_SIZE : self.current_step].copy()
        window = window.set_index('timestamp')
        return self.processor.dataframe_to_numpy(window)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        step_reward = 0
        trade_executed = False

        # 1. Gestão e Punição Severa de Stop Loss
        if config.USE_STOP_LOSS and self.shares_held > 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            if price_change <= -config.STOP_LOSS_PCT:
                action = 2 # Força a venda imediata
                step_reward += config.STRICT_STOP_LOSS_REWARD 
                if config.DEBUG: print(f"[STOP LOSS] {self.symbol} acionado em ${current_price:.2f}")

        # 2. Execução Lógica das Ações
        if action == 1 and self.balance > 0: # BUY
            cost = self.balance * config.COMMISSION
            self.shares_held = (self.balance - cost) / current_price
            self.balance = 0
            self.entry_price = current_price
            trade_executed = True
            
        elif action == 2 and self.shares_held > 0: # SELL
            sale = self.shares_held * current_price
            realized_pnl = (current_price - self.entry_price) / self.entry_price
            
            # --- MUDANÇA CRUCIAL: Recompensa baseada no PnL REALIZADO ---
            if realized_pnl > 0:
                step_reward += realized_pnl * config.PROFIT_WEIGHT # Recompensa forte por lucro real
            else:
                step_reward += realized_pnl * config.LOSS_WEIGHT   # Punição agressiva por prejuízo

            self.balance = sale * (1 - config.COMMISSION)
            self.shares_held = 0
            trade_executed = True

        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # 3. Punições Passivas (Segurar moeda caindo sem executar venda)
        if self.shares_held > 0:
            floating_pnl = (current_price - self.entry_price) / self.entry_price
            if floating_pnl < 0:
                step_reward += config.HOLDING_PENALTY 

        # 4. Taxa Fixa por Movimentação (Inibe o Overtrading)
        if trade_executed: 
            step_reward -= config.TRADE_PENALTY

        # 5. Bônus por consistência patrimonial
        if self.net_worth > config.INITIAL_BALANCE: 
            step_reward += config.CONSISTENCY_BONUS

        self.prev_net_worth = self.net_worth
        return self._get_observation(), step_reward, terminated, truncated, {}
