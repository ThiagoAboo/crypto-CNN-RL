import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import time

st.set_page_config(page_title="AI Crypto Analytics", layout="wide", page_icon="📈")

def load_data(file_name):
    path = os.path.join('bot', 'data', file_name)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except: return None
    return None

def calculate_stats(df_signals):
    """Calcula PnL real e Win Rate baseando-se em pares de ordens (Buy -> Sell)."""
    stats = {}
    for symbol in df_signals['symbol'].unique():
        symbol_df = df_signals[df_signals['symbol'] == symbol].sort_values('timestamp')
        
        trades = []
        buy_price = None
        
        # Filtra sinais de execução (1=Buy, 2=Sell)
        for _, row in symbol_df.iterrows():
            action = str(row['action'])
            if "1" in action or "BUY" in action.upper():
                buy_price = row['price']
            elif ("2" in action or "SELL" in action.upper()) and buy_price is not None:
                pnl = (row['price'] - buy_price) / buy_price
                trades.append(pnl)
                buy_price = None # Reseta para próxima operação
        
        if trades:
            wins = len([t for t in trades if t > 0])
            win_rate = (wins / len(trades)) * 100
            total_pnl = sum(trades) * 100
            stats[symbol] = {"win_rate": win_rate, "pnl": total_pnl, "total_trades": len(trades)}
        else:
            stats[symbol] = {"win_rate": 0, "pnl": 0, "total_trades": 0}
    return stats

def main():
    st.title("📊 IA Crypto - Performance & Assertividade")
    
    df_wallet = load_data('wallet_history.csv')
    df_signals = load_data('live_history.csv')

    if df_wallet is not None and not df_wallet.empty:
        df_wallet['timestamp'] = pd.to_datetime(df_wallet['timestamp'])
        latest = df_wallet.iloc[-1]
        
        # 1. Métricas de Cabeçalho
        c1, c2, c3 = st.columns(3)
        c1.metric("Patrimônio Líquido", f"${latest['net_worth']:.2f}")
        c2.metric("Saldo USD", f"${latest['balance']:.2f}")
        c3.metric("Tempo Online", f"{len(df_wallet)} min")

        # 2. Gráfico de Evolução
        st.plotly_chart(px.line(df_wallet, x='timestamp', y='net_worth', title="Curva de Equidade"), width='stretch')

    if df_signals is not None and not df_signals.empty:
        df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp'])
        stats = calculate_stats(df_signals)

        # 3. Cards de Performance por Moeda
        st.subheader("🎯 Assertividade por Ativo")
        cols = st.columns(len(stats))
        for i, (symbol, data) in enumerate(stats.items()):
            with cols[i]:
                color = "green" if data['pnl'] >= 0 else "red"
                st.markdown(f"""
                <div style="padding:15px; border-radius:10px; background-color:#1E1E1E; border-left: 5px solid {color};">
                    <h3 style="margin:0;">{symbol}</h3>
                    <p style="font-size:24px; font-weight:bold; color:{color}; margin:0;">{data['pnl']:.2f}% PnL</p>
                    <p style="margin:0;">Taxa de Acerto: <b>{data['win_rate']:.1f}%</b></p>
                    <p style="font-size:12px; color:gray;">Trades: {data['total_trades']}</p>
                </div>
                """, unsafe_allow_html=True)

        # 4. Gráfico de Comparação de PnL
        st.subheader("📊 Comparativo de Lucro/Perda")
        pnl_df = pd.DataFrame([{"Moeda": s, "PnL %": d['pnl']} for s, d in stats.items()])
        st.plotly_chart(px.bar(pnl_df, x='Moeda', y='PnL %', color='PnL %', 
                               color_continuous_scale='RdYlGn'), width='stretch')

    time.sleep(15)
    st.rerun()

if __name__ == "__main__":
    main()
