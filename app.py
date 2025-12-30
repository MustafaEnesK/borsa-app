import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import feedparser
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="NEÜ Borsa Simülasyonu", page_icon="📈", layout="wide")

# --- FONKSİYONLAR ---
@st.cache_data(ttl=600) 
def verileri_getir(sembol):
    if not sembol.endswith(".IS"):
        sembol += ".IS"
    df = yf.download(sembol, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

@st.cache_data(ttl=3600)
def temel_bilgileri_getir(sembol):
    if not sembol.endswith(".IS"):
        sembol += ".IS"
    try:
        hisse = yf.Ticker(sembol)
        info = hisse.info
        return {
            "F/K": info.get("trailingPE", "Yok"),
            "PD/DD": info.get("priceToBook", "Yok"),
            "Temettü": info.get("dividendYield", 0)
        }
    except:
        return None

def prophet_tahmin(df, gun_sayisi=30):
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=gun_sayisi)
    forecast = m.predict(future)
    return forecast

def haberleri_getir(saf_kod):
    try:
        rss_url = f"https://news.google.com/rss/search?q={saf_kod}+hisse&hl=tr&gl=TR&ceid=TR:tr"
        feed = feedparser.parse(rss_url)
        return feed.entries[:4]
    except:
        return []

# --- YAN MENÜ TASARIMI ---
with st.sidebar:
    # 1. KİŞİSEL KART
    st.markdown("""
    <div style="background-color:#1E1E1E; padding:15px; border-radius:10px; border:1px solid #333;">
        <h3 style="color:white; margin:0;">Mustafa Enes KORKMAZOĞLU</h3>
        <p style="color:#888; margin:5px 0;">🎓 NEÜ İktisat 3. Sınıf</p>
        <p style="color:#00CC96; font-weight:bold;">🚀 Borsa Simülasyonu PRO</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. İLETİŞİM BUTONLARI (YENİLENDİ)
    st.write("") 
    st.caption("İletişim & Bağlantılar")
    # Butonları alt alta ve geniş (mobilde rahat basılsın diye) ekliyoruz
    st.link_button("👔 LinkedIn Profilim", "https://www.linkedin.com/in/mustafa-enes-korkmazoglu", use_container_width=True)
    st.link_button("📸 Instagram: @mustafaenesk_", "https://www.instagram.com/mustafaenesk_", use_container_width=True)

    st.divider()

    # 3. HİSSE SEÇİMİ
    st.header("⚙️ Analiz Ayarları")
    hisse_kodu = st.text_input("Hisse Kodu", "THYAO").upper()
    tahmin_gun = st.slider("Tahmin (Gün)", 7, 90, 30)
    
    # 4. GÖRSEL AYARLAR
    st.subheader("🎨 Grafik Seçenekleri")
    goster_sma50 = st.checkbox("SMA 50 (Turuncu)", value=True)
    goster_sma200 = st.checkbox("SMA 200 (Mavi)", value=True)
    goster_bollinger = st.checkbox("Bollinger Bantları", value=False)
    
    st.divider()
    analiz_butonu = st.button("Analizi Başlat 🚀", type="primary", use_container_width=True)

# --- ANA SAYFA ---
st.title("📈 Borsa İstanbul Yapay Zeka Analisti")
st.caption(f"Hoş geldin Mustafa Enes. Piyasa verileri hazırlanıyor...")

if analiz_butonu:
    st.cache_data.clear()
    saf_kod = hisse_kodu.replace(".IS", "")
    
    with st.spinner('Yapay zeka verileri işliyor...'):
        df = verileri_getir(hisse_kodu)
        info = temel_bilgileri_getir(hisse_kodu)
        
        if df.empty:
            st.error("Veri bulunamadı!")
        else:
            # ÜST KARTLAR
            son = df['Close'].iloc[-1]
            degisim = ((son - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fiyat", f"{son:.2f} ₺", f"%{degisim:.2f}")
            if info:
                temettu = info['Temettü']
                fmt_temettu = f"%{temettu*100:.2f}" if isinstance(temettu, float) else "-"
                c2.metric("F/K", f"{info['F/K']}")
                c3.metric("PD/DD", f"{info['PD/DD']}")
                c4.metric("Temettü", fmt_temettu)

            # --- GRAFİK ÇİZİMİ ---
            st.subheader(f"📊 {saf_kod} Teknik Analiz")
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'))
            
            if goster_sma50:
                df['SMA50'] = ta.sma(df['Close'], length=50)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], line=dict(color='orange', width=1), name='SMA 50'))
            
            if goster_sma200:
                df['SMA200'] = ta.sma(df['Close'], length=200)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA200'], line=dict(color='blue', width=1), name='SMA 200'))
            
            if goster_bollinger:
                bb = ta.bbands(df['Close'], length=20, std=2)
                if bb is not None:
                    fig.add_trace(go.Scatter(x=df['Date'], y=bb[bb.columns[0]], line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dot'), name='Alt Bant'))
                    fig.add_trace(go.Scatter(x=df['Date'], y=bb[bb.columns[2]], line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', name='Üst Bant'))

            fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # TAHMİN BÖLÜMÜ
            st.subheader(f"🔮 Gelecek {tahmin_gun} Günlük Tahmin")
            try:
                fcast = prophet_tahmin(df, tahmin_gun)
                f_fig = go.Figure()
                f_fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Gerçek', line=dict(color='white')))
                future_part = fcast[fcast['ds'] > df['Date'].iloc[-1]]
                f_fig.add_trace(go.Scatter(x=future_part['ds'], y=future_part['yhat'], name='Tahmin', line=dict(color='#00CC96', dash='dash')))
                f_fig.add_trace(go.Scatter(x=future_part['ds'], y=future_part['yhat_lower'], showlegend=False, line=dict(width=0)))
                f_fig.add_trace(go.Scatter(x=future_part['ds'], y=future_part['yhat_upper'], fill='tonexty', fillcolor='rgba(0, 204, 150, 0.2)', name='Güven Aralığı', line=dict(width=0)))
                f_fig.update_layout(height=400, template="plotly_dark")
                st.plotly_chart(f_fig, use_container_width=True)
            except:
                st.warning("Tahmin hatası.")

            # HABERLER
            st.divider()
            st.subheader("📰 Güncel Haberler")
            cols = st.columns(2)
            news = haberleri_getir(saf_kod)
            if news:
                for i, h in enumerate(news):
                    cols[i % 2].info(f"[{h.title}]({h.link})")
            else:
                st.write("Haber bulunamadı.")
