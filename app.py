import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import feedparser
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Borsa PRO", page_icon="📈", layout="wide")

# --- FONKSİYONLAR ---
@st.cache_data
def verileri_getir(sembol):
    if not sembol.endswith(".IS"):
        sembol += ".IS"
    try:
        # Son 2 yılın verisi (Tahmin için uzun veri lazım)
        df = yf.download(sembol, period="2y", interval="1d", progress=False)
        # MultiIndex düzeltmesi
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def temel_bilgileri_getir(sembol):
    if not sembol.endswith(".IS"):
        sembol += ".IS"
    try:
        hisse = yf.Ticker(sembol)
        info = hisse.info
        return {
            "F/K": info.get("trailingPE", "Yok"),
            "PD/DD": info.get("priceToBook", "Yok"),
            "Temettü Verimi": info.get("dividendYield", 0),
            "Sektör": info.get("sector", "Bilinmiyor"),
            "Özet": info.get("longBusinessSummary", "Bilgi yok.")
        }
    except:
        return None

def prophet_tahmin(df, gun_sayisi=30):
    # Prophet için veri hazırlığı (ds: tarih, y: fiyat)
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Modeli Kur ve Eğit
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)
    
    # Gelecek tarihleri oluştur
    future = m.make_future_dataframe(periods=gun_sayisi)
    forecast = m.predict(future)
    
    return forecast

def haberleri_getir(saf_kod):
    try:
        rss_url = f"https://news.google.com/rss/search?q={saf_kod}+hisse&hl=tr&gl=TR&ceid=TR:tr"
        feed = feedparser.parse(rss_url)
        return feed.entries[:5]
    except:
        return []

# --- ARAYÜZ ---
st.title("📈 Borsa PRO Analist v2.0")
st.caption("Teknik Analiz | Temel Veriler | Yapay Zeka Tahmini")

# Yan Menü
with st.sidebar:
    st.header("Ayarlar")
    hisse_kodu = st.text_input("Hisse Kodu (Örn: THYAO)", "THYAO").upper()
    tahmin_gun = st.slider("Kaç Günlük Tahmin?", 7, 90, 30)
    analiz_butonu = st.button("Analizi Başlat 🚀", type="primary")

if analiz_butonu:
    saf_kod = hisse_kodu.replace(".IS", "")
    
    with st.spinner('Veriler çekiliyor, yapay zeka düşünüyor...'):
        # 1. Verileri Al
        df = verileri_getir(hisse_kodu)
        info = temel_bilgileri_getir(hisse_kodu)
        
        if df.empty:
            st.error("Veri bulunamadı!")
        else:
            # --- ÜST BİLGİ KARTLARI (TEMEL ANALİZ) ---
            son_fiyat = df['Close'].iloc[-1]
            onceki_fiyat = df['Close'].iloc[-2]
            degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Son Fiyat", f"{son_fiyat:.2f} ₺", f"%{degisim:.2f}")
            
            if info:
                fk = info['F/K']
                pddd = info['PD/DD']
                temettu = info['Temettü Verimi']
                
                col2.metric("F/K Oranı", f"{fk if fk == 'Yok' else f'{fk:.2f}'}")
                col3.metric("PD/DD", f"{pddd if pddd == 'Yok' else f'{pddd:.2f}'}")
                col4.metric("Temettü", f"%{temettu*100:.2f}" if isinstance(temettu, float) else "Yok")
            
            # --- İNTERAKTİF GRAFİK (PLOTLY) ---
            st.subheader("📊 İnteraktif Teknik Grafik")
            
            # İndikatörler
            df['SMA50'] = ta.sma(df['Close'], length=50)
            df['SMA200'] = ta.sma(df['Close'], length=200)
            
            fig = go.Figure()
            
            # Mum Grafiği
            fig.add_trace(go.Candlestick(x=df['Date'],
                            open=df['Open'], high=df['High'],
                            low=df['Low'], close=df['Close'], name='Fiyat'))
            
            # Ortalamalar
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], line=dict(color='orange', width=1), name='SMA 50'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA200'], line=dict(color='blue', width=1), name='SMA 200'))
            
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            # --- YAPAY ZEKA TAHMİNİ (PROPHET) ---
            st.subheader(f"🔮 Gelecek {tahmin_gun} Günlük AI Tahmini")
            
            forecast = prophet_tahmin(df, tahmin_gun)
            
            # Tahmin Grafiği
            fig_tahmin = go.Figure()
            
            # Geçmiş Veri
            fig_tahmin.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Gerçekleşen', line=dict(color='white')))
            
            # Gelecek Tahmin
            gelecek_veri = forecast[forecast['ds'] > df['Date'].iloc[-1]]
            fig_tahmin.add_trace(go.Scatter(x=gelecek_veri['ds'], y=gelecek_veri['yhat'], name='Tahmin', line=dict(color='cyan', dash='dash')))
            
            # Güven Aralığı (Alt-Üst Sınır)
            fig_tahmin.add_trace(go.Scatter(x=gelecek_veri['ds'], y=gelecek_veri['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
            fig_tahmin.add_trace(go.Scatter(x=gelecek_veri['ds'], y=gelecek_veri['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 255, 0.2)', name='Güven Aralığı'))
            
            fig_tahmin.update_layout(height=400, template="plotly_dark", title="Prophet Modeli Tahmini")
            st.plotly_chart(fig_tahmin, use_container_width=True)
            
            # --- HABERLER ---
            st.divider()
            st.subheader("📰 Piyasa Haberleri")
            haberler = haberleri_getir(saf_kod)
            for h in haberler:
                st.write(f"• [{h.title}]({h.link})")
