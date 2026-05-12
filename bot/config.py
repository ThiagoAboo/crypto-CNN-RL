import os

# Configurações de Sistema
DEBUG = True
DEBUG_LOAD = False
DEBUG_TRACE = False
DEBUG_DECISION = False
DEBUG_TRAINING = False

# Configurações de Mercado
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT'] 
TIMEFRAME = '5m'
HISTORY_LIMIT = 5000

# Configurações da Imagem (CNN)
IMG_SIZE = (64, 64)
IMG_DPI = 100        
WINDOW_SIZE = 60 

# Configurações do Ambiente de Trading
INITIAL_BALANCE = 1000.0
COMMISSION = 0.00075 
ALLOCATION_PER_TRADE = 0.20  # Cada moeda usa 20% do capital inicial ($200)
MIN_BALANCE_RESERVE = 50.0   # Nunca deixa o saldo USD cair abaixo de $50
RESET_WALLET = False

# --- NOVOS PARÂMETROS DE REWARD ---
# Sugestão: Penalidade levemente maior que a taxa para evitar giros inúteis
TRADE_PENALTY = 0.001    
PROFIT_WEIGHT = 10.0     
CONSISTENCY_BONUS = 0.001 

# --- SEGURANÇA E STOP LOSS ---
USE_STOP_LOSS = True
STOP_LOSS_PCT = 0.02  # 2% de queda máxima permitida por operação

# Configurações de IA
ONLINE_LEARNING_RATE = 5e-5
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.99  
N_STEPS = 512          
TOTAL_TIMESTEPS = 50000  # Excelente valor para começar a ver padrões!

# Paths
MODEL_DIR = os.path.join('bot', 'models')
DATA_DIR = os.path.join('bot', 'data')