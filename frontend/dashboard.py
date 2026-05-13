import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import time
from datetime import datetime, timedelta
import bot.config as config  # Centralização global de parâmetros

# Configurações de layout e aba do navegador
st.set_page_config(page_title="IA Crypto Intelligence", layout="wide", page_icon="🤖")

# --- CSS CUSTOMIZADO ---
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
    [data-testid="stMetricValue"] { font-size: 24px; color: #00ffcc; }
    </style>
""", unsafe_allow_html=True)

def load_data(file_name):
    path = os.path.join(config.DATA_DIR, file_name)
    if os.path.exists(path):
        try: return pd.read_csv(path).copy()
        except: return None
    return None

def load_wallet_state():
    path = os.path.join(config.DATA_DIR, 'wallet_state.json')
    if os.path.exists(path):
        try:
            with open(path, 'r') as f: return json.load(f)
        except: return None
    return None

def calculate_detailed_stats(df_signals, wallet_state, days_filter):
    stats = {}
    if df_signals is None or df_signals.empty: return stats
    
    cutoff_date = datetime.now() - timedelta(days=days_filter)
    df_filtered = df_signals[pd.to_datetime(df_signals['timestamp']) >= cutoff_date]
    
    for symbol in config.SYMBOLS:
        symbol_df = df_filtered[df_filtered['symbol'] == symbol].sort_values('timestamp')
        trades = []
        buy_price = None
        
        for _, row in symbol_df.iterrows():
            action = str(row['action']).upper()
            price = float(row['price'])
            if "BUY" in action: 
                buy_price = price
            elif "SELL" in action and buy_price is not None:
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
    st.sidebar.title("⚙️ Filtros de Análise")
    periodo = st.sidebar.selectbox(
        "Período de Performance", 
        options=[1, 7, 30, 90],
        format_func=lambda x: f"Últimas 24h" if x==1 else f"Últimos {x} dias", 
        index=1
    )

    st.title("🤖 IA Crypto Intelligence Terminal")
    
    df_wallet = load_data('wallet_history.csv')
    df_signals = load_data('live_history.csv')
    wallet_state = load_wallet_state()

    if df_wallet is not None and not df_wallet.empty:
        latest = df_wallet.iloc[-1]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Net Worth", f"${latest['net_worth']:.2f}")
        c2.metric("Available USD", f"${latest['balance']:.2f}")
        c3.metric("Tempo Online", f"{len(df_wallet)} min")
        total_pnl_pct = ((latest['net_worth'] - config.INITIAL_BALANCE) / config.INITIAL_BALANCE) * 100
        c4.metric("Total PnL (Geral)", f"{total_pnl_pct:.2f}%")
        active_count = len([s for s in wallet_state["positions"].values() if s > 0]) if wallet_state else 0
        c5.metric("Active Assets", active_count)

    st.divider()

    stats = calculate_detailed_stats(df_signals, wallet_state, periodo)
    if stats:
        st.subheader(f"📡 Performance ({periodo}d)")
        cols = st.columns(len(stats))
        for i, (symbol, data) in enumerate(stats.items()):
            with cols[i]:
                color = "#00FF7F" if data['pnl_total'] >= 0 else "#FF4B4B"
                border = "2px solid #00FFCC" if data['is_active'] else "1px solid #2b2f36"
                st.markdown(f"""
                    <div class="asset-card" style="border: {border};">
                        <h3 style="margin:0; color:#00FFCC; font-size:16px;">{symbol}</h3>
                        <p style="font-size:22px; font-weight:bold; color:{color}; margin:5px 0;">{data['pnl_total']:.2f}%</p>
                        <div class="balance-box">
                            <p style="margin:0; font-size:9px; color:#888;">SALDO ATUAL</p>
                            <p style="margin:0; font-size:14px; font-weight:bold; color:white;">{data['qty']:.6f}</p>
                            <p style="margin:0; font-size:11px; color:#555;">≈ ${data['qty'] * data['last_price']:.2f} USD</p>
                        </div>
                        <p style="margin-top:10px; font-size:11px; color:#aaa;">Win Rate: <b>{data['win_rate']:.1f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)

    st.divider()
    g1, g2 = st.columns(2)
    
    with g1:
        st.subheader("📈 Evolução Patrimonial e Sinais da IA")
        if df_wallet is not None and not df_wallet.empty:
            df_wallet['timestamp'] = pd.to_datetime(df_wallet['timestamp'])
            fig = go.Figure()
            
            # 1. Linha principal do Patrimônio Líquido Real
            fig.add_trace(go.Scatter(
                x=df_wallet['timestamp'], y=df_wallet['net_worth'],
                mode='lines', name='Net Worth', line=dict(color='#00FFCC', width=3)
            ))
            
            # --- 2. CÁLCULO E INJEÇÃO DA MÉDIA MÓVEL EXPONENCIAL (EMA-10) ---
            # Calcula a média suavizada dos últimos 10 registros do seu patrimônio
            if len(df_wallet) >= 2:
                df_wallet['ema_patrimonio'] = df_wallet['net_worth'].ewm(span=10, adjust=False).mean()
                fig.add_trace(go.Scatter(
                    x=df_wallet['timestamp'], y=df_wallet['ema_patrimonio'],
                    mode='lines', name='Tendência (EMA 10)', 
                    line=dict(color='#FFD700', width=2, dash='dash') # Linha amarela tracejada
                ))
            
            if df_signals is not None and not df_signals.empty:
                df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp'])
                df_signals['action_str'] = df_signals['action'].astype(str).str.upper()
                
                def find_net_worth(timestamp):
                    time_diffs = (df_wallet['timestamp'] - timestamp).abs()
                    return df_wallet.loc[time_diffs.idxmin(), 'net_worth']
                
                buys_grouped = df_signals[df_signals['action_str'] == "BUY"].groupby('timestamp')['symbol'].apply(
                    lambda x: "<br>".join([f"<b>Compra: {s}</b><br>Data: %{{x}}<br>Patrimônio: $%{{y:.2f}}" for s in x])
                ).reset_index()
                
                sells_grouped = df_signals[df_signals['action_str'] == "SELL"].groupby('timestamp')['symbol'].apply(
                    lambda x: "<br>".join([f"<b>Venda: {s}</b><br>Data: %{{x}}<br>Patrimônio: $%{{y:.2f}}" for s in x])
                ).reset_index()
                
                if not buys_grouped.empty:
                    buys_grouped['net_worth'] = buys_grouped['timestamp'].apply(find_net_worth)
                    for _, row in buys_grouped.iterrows():
                        fig.add_trace(go.Scatter(
                            x=[row['timestamp']], y=[row['net_worth']], mode='markers', name='Sinal: COMPRA',
                            showlegend=False,
                            marker=dict(symbol='triangle-up', size=12, color='#00FF7F', line=dict(color='black', width=1)),
                            hovertemplate=row['symbol'] + "<extra></extra>"
                        ))
                
                if not sells_grouped.empty:
                    sells_grouped['net_worth'] = sells_grouped['timestamp'].apply(find_net_worth)
                    for _, row in sells_grouped.iterrows():
                        fig.add_trace(go.Scatter(
                            x=[row['timestamp']], y=[row['net_worth']], mode='markers', name='Sinal: VENDA',
                            showlegend=False,
                            marker=dict(symbol='triangle-down', size=12, color='#FF4B4B', line=dict(color='black', width=1)),
                            hovertemplate=row['symbol'] + "<extra></extra>"
                        ))
            
            y_min = df_wallet['net_worth'].min() * 0.9995
            y_max = df_wallet['net_worth'].max() * 1.0005
            
            fig.update_layout(
                template="plotly_dark", yaxis=dict(range=[y_min, y_max], autorange=False, title="Net Worth ($)"),
                xaxis=dict(title="Horário"), margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                legend=dict(orientation="h", y=1.1, x=0)
            )
            st.plotly_chart(fig, width='stretch')

    with g2:
        st.subheader(f"🎯 Dispersão ({periodo}d)")
        all_t = []
        if stats:
            for s, d in stats.items():
                for t in d['trades_history']: all_t.append({"Ativo": s, "Lucro %": t * 100})
        if all_t:
            fig_d = px.strip(pd.DataFrame(all_t), x="Ativo", y="Lucro %", color="Ativo", template="plotly_dark")
            fig_d.add_hline(y=0, line_dash="dash", line_color="white")
            st.plotly_chart(fig_d, width='stretch')

    time.sleep(15)
    st.rerun()

if __name__ == "__main__":
    main()
