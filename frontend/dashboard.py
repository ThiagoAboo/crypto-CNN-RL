import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import time

# Configurações de layout e aba do navegador
st.set_page_config(page_title="IA Crypto Intelligence", layout="wide", page_icon="🤖")

# --- CSS CUSTOMIZADO (Visual Dark Terminal) ---
st.markdown("""
    <style>
    .main { background-color: #0b0d10; }
    .asset-card {
        padding: 20px; border-radius: 15px; background-color: #161a1e;
        border: 1px solid #2b2f36; margin-bottom: 10px;
    }
    .balance-box {
        background: #1e2329; padding: 12px; border-radius: 8px; margin-top: 10px;
        border: 1px solid #333;
    }
    /* Estilização das métricas padrão do Streamlit */
    [data-testid="stMetricValue"] { font-size: 24px; color: #00ffcc; }
    </style>
""", unsafe_allow_html=True)

def load_data(file_name):
    """Carrega os históricos em CSV."""
    path = os.path.join('bot', 'data', file_name)
    if os.path.exists(path):
        try: return pd.read_csv(path)
        except: return None
    return None

def load_wallet_state():
    """Carrega o estado atual (JSON) para saber o saldo real das moedas."""
    path = os.path.join('bot', 'data', 'wallet_state.json')
    if os.path.exists(path):
        try:
            with open(path, 'r') as f: return json.load(f)
        except: return None
    return None

def calculate_detailed_stats(df_signals, wallet_state):
    """Calcula PnL por ativo e win rate."""
    stats = {}
    if df_signals is None or df_signals.empty: return stats
    
    for symbol in df_signals['symbol'].unique():
        symbol_df = df_signals[df_signals['symbol'] == symbol].sort_values('timestamp')
        trades = []
        buy_price = None
        
        for _, row in symbol_df.iterrows():
            action = str(row['action']).upper()
            price = float(row['price'])
            if "BUY" in action or "1" in action: 
                buy_price = price
            elif ("SELL" in action or "2" in action) and buy_price is not None:
                trades.append((price - buy_price) / buy_price)
                buy_price = None 

        qty = wallet_state["positions"].get(symbol, 0.0) if wallet_state else 0.0
        stats[symbol] = {
            "pnl_total": sum(trades) * 100,
            "win_rate": (len([t for t in trades if t > 0]) / len(trades) * 100) if trades else 0,
            "qty": qty,
            "is_active": qty > 0,
            "last_price": symbol_df.iloc[-1]['price'] if not symbol_df.empty else 0,
            "trades_history": trades
        }
    return stats

def main():
    st.title("🤖 IA Crypto Intelligence Terminal")
    
    # Carregamento de dados
    df_wallet = load_data('wallet_history.csv')
    df_signals = load_data('live_history.csv')
    wallet_state = load_wallet_state()

    # --- KPI HEADER (5 colunas com Tempo Online recuperado) ---
    if df_wallet is not None and not df_wallet.empty:
        latest = df_wallet.iloc[-1]
        c1, c2, c3, c4, c5 = st.columns(5)
        
        c1.metric("Net Worth", f"${latest['net_worth']:.2f}")
        c2.metric("Available USD", f"${latest['balance']:.2f}")
        c3.metric("Tempo Online", f"{len(df_wallet)} min")
        c4.metric("Total PnL", f"{((latest['net_worth'] - 1000)/10):.2f}%")
        
        active_count = len([s for s in wallet_state["positions"].values() if s > 0]) if wallet_state else 0
        c5.metric("Active Assets", active_count)

    st.divider()

    # --- SEÇÃO DE CARDS DE MONITORAMENTO ---
    stats = calculate_detailed_stats(df_signals, wallet_state)
    if stats:
        st.subheader("📡 Monitoramento de Ativos")
        cols = st.columns(len(stats))
        for i, (symbol, data) in enumerate(stats.items()):
            with cols[i]:
                color = "#00FF7F" if data['pnl_total'] >= 0 else "#FF4B4B"
                border = "2px solid #00FFCC" if data['is_active'] else "1px solid #2b2f36"
                
                st.markdown(f"""
                    <div class="asset-card" style="border: {border};">
                        <h3 style="margin:0; color:#00FFCC; font-size:18px;">{symbol}</h3>
                        <p style="font-size:26px; font-weight:bold; color:{color}; margin:5px 0;">{data['pnl_total']:.2f}%</p>
                        <div class="balance-box">
                            <p style="margin:0; font-size:10px; color:#888;">SALDO DA MOEDA</p>
                            <p style="margin:0; font-size:15px; font-weight:bold; color:white;">{data['qty']:.6f}</p>
                            <p style="margin:0; font-size:11px; color:#555;">≈ ${data['qty'] * data['last_price']:.2f} USD</p>
                        </div>
                        <p style="margin-top:10px; font-size:12px; color:#aaa;">Win Rate: <b>{data['win_rate']:.1f}%</b></p>
                        <p style="margin:0; font-size:10px; color:#666;">Status: {'🟢 EM POSIÇÃO' if data['is_active'] else '⚪ AGUARDANDO'}</p>
                    </div>
                """, unsafe_allow_html=True)

    # --- SEÇÃO DE GRÁFICOS (Evolução com Zoom e Dispersão) ---
    st.divider()
    g1, g2 = st.columns(2)
    
    with g1:
        st.subheader("📈 Evolução Patrimonial")
        if df_wallet is not None and not df_wallet.empty:
            df_wallet['timestamp'] = pd.to_datetime(df_wallet['timestamp'])
            
            fig = px.line(df_wallet, x='timestamp', y='net_worth', template="plotly_dark")
            fig.update_traces(line_color='#00FFCC', line_width=3)
            
            # --- AJUSTE DE ESCALA CIRÚRGICO ---
            # Define o zoom baseado no min/max real com uma folga de 0.05%
            y_min = df_wallet['net_worth'].min() * 0.9995
            y_max = df_wallet['net_worth'].max() * 1.0005
            
            fig.update_layout(
                yaxis=dict(range=[y_min, y_max], autorange=False, title="Net Worth ($)"),
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

    with g2:
        st.subheader("🎯 Dispersão de Lucros")
        all_t = []
        if stats:
            for s, d in stats.items():
                for t in d['trades_history']: 
                    all_t.append({"Ativo": s, "Lucro %": t * 100})
        
        if all_t:
            fig_d = px.strip(pd.DataFrame(all_t), x="Ativo", y="Lucro %", color="Ativo", template="plotly_dark")
            fig_d.add_hline(y=0, line_dash="dash", line_color="white")
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("Aguardando finalização de ciclos (Compra/Venda) para gerar dispersão.")

    # Auto-refresh a cada 15 segundos
    time.sleep(15)
    st.rerun()

if __name__ == "__main__":
    main()
