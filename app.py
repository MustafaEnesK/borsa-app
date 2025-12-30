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
# DİKKAT: Cache süresini kısalttık ve hata yakalamayı kaldırdık ki sorunu görelim
@st.cache_data(ttl=600) 
def verileri_getir(sembol):
    if not sembol.endswith(".IS"):
        sembol += ".IS"
    
    # auto_adjust=True parametresi EKLENDİ (Önemli!)
    df = yf.download(sembol, period="2y", interval="1d", progress=False, auto_adjust=True)
    
    # Veri geldi mi kontrolü
    if df.empty:
        return pd.DataFrame()

    # MultiIndex düzeltmesi (Sütunlar bazen ('Close', 'THYAO.IS') gibi gelir)
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
    # Prophet sütun isimlerini 'ds' ve 'y' ister
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

# --- ARAYÜZ ---
st.title("📈 Borsa PRO Analist v2.1")
st.caption("Teknik Analiz | Temel Veriler | Yapay Zeka Tahmini")

# Yan Menü
with st.sidebar:
    st.header("Ayarlar")
    hisse_kodu = st.text_input("Hisse Kodu (Örn: THYAO)", "THYAO").upper()
    tahmin_gun = st.slider("Kaç Günlük Tahmin?", 7, 90, 30)
    analiz_butonu = st.button("Analizi Başlat 🚀", type="primary")

if analiz_butonu:
    # Kullanıcı "Analiz Et"e basınca Cache'i temizle (Manuel Refresh)
    st.cache_data.clear()
    
    saf_kod = hisse_kodu.replace(".IS", "")
    
    with st.spinner(f'{saf_kod} verileri ve yapay zeka analizi hazırlanıyor...'):
        df = verileri_getir(hisse_kodu)
        info = temel_bilgileri_getir(hisse_kodu)
        
        if df.empty:
            st.error(f"❌ '{hisse_kodu}' için veri bulunamadı! Lütfen kodu kontrol edin (Örn: GARAN, ASELS).")
        else:
            # --- ÜST BİLGİ KARTLARI ---
            son_fiyat = df['Close'].iloc[-1]
            # Veri setinde en az 2 gün var mı kontrolü
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
                
                # Temettü None gelirse hata vermesin
                temettu_yazi = "Yok"
                if temettu and isinstance(temettu, (int, float)):
                    temettu_yazi = f"%{temettu*100:.2f}"
                
                col2.metric("F/K Oranı", f"{fk if fk != 'Yok' and fk is not None else 'Yok'}")
                col3.metric("PD/DD", f"{pddd if pddd != 'Yok' and pddd is not None else 'Yok'}")
                col4.metric("Temettü", temettu_yazi)
            
            # --- İNTERAKTİF GRAFİK (PLOTLY) ---
            st.subheader("📊 İnteraktif Teknik Grafik")
            
            df['SMA50'] = ta.sma(df['Close'], length=50)
            df['SMA200'] = ta.sma(df['Close'], length=200)
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df['Date'],
                            open=df['Open'], high=df['High'],
                            low=df['Low'], close=df['Close'], name='Fiyat'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], line=dict(color='orange', width=1), name='SMA 50'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA200'], line=dict(color='blue', width=1), name='SMA 200'))
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            # --- YAPAY ZEKA TAHMİNİ ---
            st.subheader(f"🔮 Gelecek {tahmin_gun} Günlük AI Tahmini (Prophet)")
            try:
                forecast = prophet_tahmin(df, tahmin_gun)
                
                fig_tahmin = go.Figure()
                fig_tahmin.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Gerçekleşen', line=dict(color='white')))
                
                gelecek_veri = forecast[forecast['ds'] > df['Date'].iloc[-1]]
                fig_tahmin.add_trace(go.Scatter(x=gelecek_veri['ds'], y=gelecek_veri['yhat'], name='Tahmin', line=dict(color='cyan', dash='dash')))
                fig_tahmin.add_trace(go.Scatter(x=gelecek_veri['ds'], y=gelecek_veri['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
                fig_tahmin.add_trace(go.Scatter(x=gelecek_veri['ds'], y=gelecek_veri['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 255, 0.2)', name='Güven Aralığı'))
                fig_tahmin.update_layout(height=400, template="plotly_dark", title="Trend Tahmini")
                st.plotly_chart(fig_tahmin, use_container_width=True)
            except Exception as e:
                st.warning(f"Tahmin oluşturulurken hata: {e}")
            
            # --- HABERLER ---
            st.divider()
            st.subheader("📰 Piyasa Haberleri")
            haberler = haberleri_getir(saf_kod)
            if haberler:
                for h in haberler:
                    st.write(f"• [{h.title}]({h.link})")
            else:
                st.info("Haber bulunamadı.")
