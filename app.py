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
    
    # auto_adjust=True ile hisse bölünmelerini hesaba katar
    df = yf.download(sembol, period="2y", interval="1d", progress=False, auto_adjust=True)
    
    if df.empty:
        return pd.DataFrame()

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
            "Temettü Verimi": info.get("dividendYield", 0),
            "Sektör": info.get("sector", "Bilinmiyor"),
            "Özet": info.get("longBusinessSummary", "Bilgi yok.")
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
        return feed.entries[:5]
    except:
        return []

# --- ARAYÜZ BAŞLIYOR ---

# YAN MENÜ (SOL TARAFTAKİ KISIM)
with st.sidebar:
    # --- İMZA KISMI (Burayı Ekledik) ---
    st.info("👨‍💻 **Geliştirici: Mustafa Enes KORKMAZOĞLU**\n\n🎓 NEÜ İktisat 3. Sınıf\n\n🚀 Borsa Simülasyonu DEMO")
    
    st.divider() # Araya çizgi çeker
    
    st.header("⚙️ Ayarlar")
    hisse_kodu = st.text_input("Hisse Kodu (Örn: THYAO)", "THYAO").upper()
    tahmin_gun = st.slider("Tahmin Süresi (Gün)", 7, 90, 30)
    analiz_butonu = st.button("Analizi Başlat 🚀", type="primary")
    
    st.divider()
    st.caption("Not: Veriler yfinance üzerinden çekilmektedir ve 15dk gecikmeli olabilir.")

# ANA SAYFA
st.title("📈 Borsa İstanbul Yapay Zeka Analisti")
st.caption(f"Hoş geldin Mustafa Enes. Bugün piyasalar nasıl?")

if analiz_butonu:
    # Yeni analizde cache temizle
    st.cache_data.clear()
    
    saf_kod = hisse_kodu.replace(".IS", "")
    
    with st.spinner(f'{saf_kod} analiz ediliyor...'):
        df = verileri_getir(hisse_kodu)
        info = temel_bilgileri_getir(hisse_kodu)
        
        if df.empty:
            st.error(f"❌ '{hisse_kodu}' verisi alınamadı! BIST kodunu kontrol et.")
        else:
            # 1. BİLGİ KARTLARI
            son_fiyat = df['Close'].iloc[-1]
            if len(df) > 1:
                onceki_fiyat = df['Close'].iloc[-2]
                degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
            else:
                degisim = 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Son Fiyat", f"{son_fiyat:.2f} ₺", f"%{degisim:.2f}")
            
            if info:
                fk = info['F/K']
                pddd = info['PD/DD']
                temettu = info['Temettü Verimi']
                
                temettu_yazi = "Yok"
                if temettu and isinstance(temettu, (int, float)):
                    temettu_yazi = f"%{temettu*100:.2f}"
                
                col2.metric("F/K", f"{fk if fk != 'Yok' and fk is not None else '-'}")
                col3.metric("PD/DD", f"{pddd if pddd != 'Yok' and pddd is not None else '-'}")
                col4.metric("Temettü", temettu_yazi)
            
            # 2. GRAFİK
            st.subheader(f"📊 {saf_kod} Teknik Görünüm")
            df['SMA50'] = ta.sma(df['Close'], length=50)
            df['SMA200'] = ta.sma(df['Close'], length=200)
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], line=dict(color='orange', width=1), name='SMA 50'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA200'], line=dict(color='blue', width=1), name='SMA 200'))
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            # 3. TAHMİN
            st.subheader(f"🔮 Gelecek {tahmin_gun} Günlük Tahmin (Prophet AI)")
            try:
                forecast = prophet_tahmin(df, tahmin_gun)
                fig_tahmin = go.Figure()
                fig_tahmin.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Gerçekleşen', line=dict(color='white')))
                gelecek_veri = forecast[forecast['ds'] > df['Date'].iloc[-1]]
                fig_tahmin.add_trace(go.Scatter(x=gelecek_veri['ds'], y=gelecek_veri['yhat'], name='Tahmin', line=dict(color='cyan', dash='dash')))
                fig_tahmin.add_trace(go.Scatter(x=gelecek_veri['ds'], y=gelecek_veri['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 255, 0.2)', name='Güven Aralığı'))
                fig_tahmin.update_layout(height=400, template="plotly_dark")
                st.plotly_chart(fig_tahmin, use_container_width=True)
            except:
                st.warning("Tahmin oluşturulamadı.")
            
            # 4. HABERLER
            st.divider()
            st.subheader("📰 İlgili Haberler")
            haberler = haberleri_getir(saf_kod)
            if haberler:
                for h in haberler:
                    st.markdown(f"**[{h.title}]({h.link})**")
            else:
                st.info("Güncel haber bulunamadı.")
