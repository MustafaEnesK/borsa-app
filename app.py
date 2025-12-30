import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import feedparser
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import cvxpy as cp
import numpy as np
import statsmodels.api as sm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="BorsApp - Pro Terminal", page_icon="📈", layout="wide")

# --- YASAL UYARI (SAFETY GATE) ---
if 'yasal_kabul' not in st.session_state: st.session_state.yasal_kabul = False

if not st.session_state.yasal_kabul:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="background-color:#262730; padding:30px; border-radius:15px; border:2px solid #FF4B4B; text-align:center;">
            <h1 style="color:#FF4B4B;">⚠️ YASAL UYARI</h1>
            <hr>
            <p>Bu uygulama eğitim amaçlıdır. <strong>Yatırım tavsiyesi değildir.</strong></p>
            <p>Yapay zeka verileri yanılabilir. Tüm risk kullanıcıya aittir.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("✅ Kabul Ediyorum", type="primary", use_container_width=True):
            st.session_state.yasal_kabul = True
            st.rerun()
    st.stop()

# --- LİSTELER VE AYARLAR ---
if 'analiz_aktif' not in st.session_state: st.session_state.analiz_aktif = False
if 'secilen_hisse' not in st.session_state: st.session_state.secilen_hisse = ""

@st.cache_data(ttl=86400)
def listeleri_hazirla():
    bist30 = sorted(["AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "DOAS", "EKGYO", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", "KOZAL", "KRDMD", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK"])
    return bist30, bist30 # Demo için kısaltıldı, normalde full liste buraya gelir

BIST_30_LISTESI, _ = listeleri_hazirla()

# --- FONKSİYONLAR ---
@st.cache_data(ttl=600) 
def veri_cek(kod):
    if not kod.endswith(".IS"): kod += ".IS"
    df = yf.download(kod, period="1y", interval="1d", progress=False, auto_adjust=True)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

def sirket_ismini_bul(kod):
    """Hisse kodundan şirketin tam adını bulur."""
    try:
        if not kod.endswith(".IS"): kod += ".IS"
        ticker = yf.Ticker(kod)
        return ticker.info.get('longName', kod)
    except:
        return kod

def haberleri_getir(kod):
    try:
        rss_url = f"https://news.google.com/rss/search?q={kod}+hisse&hl=tr&gl=TR&ceid=TR:tr"
        feed = feedparser.parse(rss_url)
        return feed.entries[:6]
    except: return []

# --- AI MODELLERİ ---
def xgboost_analiz(df):
    data = df.copy()
    try:
        data['RSI'] = ta.rsi(data['Close'], 14)
        data['SMA'] = ta.sma(data['Close'], 50)
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        data.dropna(inplace=True)
        
        X = data[['RSI', 'SMA', 'Volume']]
        y = data['Target']
        X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, shuffle=False)
        
        model = XGBClassifier(eval_metric='logloss')
        model.fit(X_train, y_train)
        
        last = X.iloc[[-1]]
        pred = model.predict(last)[0]
        prob = model.predict_proba(last)[0]
        return pred, prob
    except: return None, None

def prophet_tahmin(df, gun_sayisi=30):
    try:
        df_p = df[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
        m = Prophet()
        m.fit(df_p)
        future = m.make_future_dataframe(periods=gun_sayisi)
        forecast = m.predict(future)
        return forecast
    except: return pd.DataFrame()

# --- ARAYÜZ ---
with st.sidebar:
    st.title("BorsApp 🚀")
    st.caption("Professional AI Terminal")
    st.divider()
    sayfa = st.radio("Menü", ["🔎 Hisse Analiz Terminali", "🏆 Canlı Piyasa", "⚖️ Portföy"])

# --- SAYFA: HİSSE ANALİZ TERMİNALİ ---
if sayfa == "🔎 Hisse Analiz Terminali":
    
    # ARAMA ÇUBUĞU
    col_search1, col_search2 = st.columns([3, 1])
    with col_search1:
        kod_giris = st.text_input("Hisse Kodu Giriniz (Örn: ASELS, THYAO)", "THYAO").upper()
    with col_search2:
        st.write("")
        st.write("")
        btn_analiz = st.button("ANALİZ ET 🔍", type="primary", use_container_width=True)

    if btn_analiz or kod_giris:
        with st.spinner("Yapay Zeka verileri işliyor..."):
            df = veri_cek(kod_giris)
            
            if not df.empty:
                # Şirket İsmi ve Başlık
                tam_isim = sirket_ismini_bul(kod_giris)
                son_fiyat = df['Close'].iloc[-1]
                degisim = ((son_fiyat - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                renk_degisim = "green" if degisim > 0 else "red"
                
                # BAŞLIK KISMI
                st.markdown(f"""
                ### {tam_isim} ({kod_giris})
                <h1 style='margin-top:-15px;'>{son_fiyat:.2f} ₺ <span style='color:{renk_degisim}; font-size:24px;'>%{degisim:.2f}</span></h1>
                """, unsafe_allow_html=True)
                
                # SEKMELER
                tab_genel, tab_ai, tab_haber, tab_gecmis = st.tabs([
                    "📊 Genel Bakış", 
                    "🧠 Yapay Zeka & Tahmin", 
                    "📰 Haberler", 
                    "📅 Son 1 Ay Verisi"
                ])
                
                # 1. TAB: GENEL BAKIŞ
                with tab_genel:
                    # Grafik
                    fig = go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat")])
                    fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Temel Veriler (Hızlı Özet)
                    try:
                        info = yf.Ticker(kod_giris+".IS").info
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("F/K Oranı", f"{info.get('trailingPE', '-'):.2f}" if info.get('trailingPE') else "-")
                        c2.metric("PD/DD", f"{info.get('priceToBook', '-'):.2f}" if info.get('priceToBook') else "-")
                        c3.metric("Hacim (Lot)", f"{df['Volume'].iloc[-1]:,}")
                        c4.metric("52 Hafta En Yüksek", f"{info.get('fiftyTwoWeekHigh', '-')} ₺")
                    except: st.warning("Temel veriler alınamadı.")

                # 2. TAB: YAPAY ZEKA & TAHMİN
                with tab_ai:
                    col_ai_L, col_ai_R = st.columns([1, 2])
                    
                    # XGBoost Sinyali
                    pred, prob = xgboost_analiz(df)
                    
                    with col_ai_L:
                        st.subheader("🤖 AI Yorumu")
                        if pred is not None:
                            guven = max(prob) * 100
                            if pred == 1:
                                st.success("YÖN: YÜKSELİŞ (AL)")
                                st.write(f"Yapay zeka, teknik indikatörlere bakarak **%{guven:.1f}** güvenle yükseliş öngörüyor.")
                            else:
                                st.error("YÖN: DÜŞÜŞ (SAT/BEKLE)")
                                st.write(f"Yapay zeka, teknik indikatörlere bakarak **%{guven:.1f}** güvenle düşüş veya düzeltme öngörüyor.")
                            
                            # İndikatör Detayı
                            rsi = ta.rsi(df['Close'], 14).iloc[-1]
                            st.info(f"💡 **AI İpucu:** RSI şu an **{rsi:.1f}** seviyesinde.")
                        else:
                            st.warning("Yeterli veri olmadığı için AI yorum yapamıyor.")

                    # Prophet Tahmini
                    with col_ai_R:
                        st.subheader("🔮 Gelecek Tahmin Grafiği (Prophet)")
                        forecast = prophet_tahmin(df, 60) # 60 Günlük
                        if not forecast.empty:
                            fig_p = go.Figure()
                            # Geçmiş
                            fig_p.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Gerçekleşen', line=dict(color='white', width=1)))
                            # Gelecek
                            future_data = forecast[forecast['ds'] > df['Date'].iloc[-1]]
                            fig_p.add_trace(go.Scatter(x=future_data['ds'], y=future_data['yhat'], name='AI Tahmini', line=dict(color='#00CC96', width=2, dash='dash')))
                            # Aralık
                            fig_p.add_trace(go.Scatter(x=future_data['ds'], y=future_data['yhat_upper'], showlegend=False, line=dict(width=0)))
                            fig_p.add_trace(go.Scatter(x=future_data['ds'], y=future_data['yhat_lower'], fill='tonexty', fillcolor='rgba(0, 204, 150, 0.2)', name='Güven Aralığı', line=dict(width=0)))
                            
                            fig_p.update_layout(height=350, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
                            st.plotly_chart(fig_p, use_container_width=True)

                # 3. TAB: HABERLER
                with tab_haber:
                    st.subheader(f"📰 {tam_isim} Haber Akışı")
                    haberler = haberleri_getir(kod_giris)
                    if haberler:
                        for h in haberler:
                            with st.container():
                                st.markdown(f"**[{h.title}]({h.link})**")
                                st.caption(f"Kaynak: {h.source.title} • {h.published}")
                                st.divider()
                    else:
                        st.info("Güncel haber bulunamadı.")

                # 4. TAB: SON 1 AY VERİSİ
                with tab_gecmis:
                    st.subheader("📅 Son 30 Günlük İşlem Verileri")
                    df_last = df.tail(22).sort_values(by="Date", ascending=False) # İş günü olarak 22 gün ~ 1 Ay
                    
                    # Tabloyu güzelleştir
                    st.dataframe(
                        df_last[['Open', 'High', 'Low', 'Close', 'Volume']],
                        column_config={
                            "Open": st.column_config.NumberColumn("Açılış", format="%.2f"),
                            "High": st.column_config.NumberColumn("Yüksek", format="%.2f"),
                            "Low": st.column_config.NumberColumn("Düşük", format="%.2f"),
                            "Close": st.column_config.NumberColumn("Kapanış", format="%.2f"),
                            "Volume": st.column_config.NumberColumn("Hacim", format="%d"),
                        },
                        use_container_width=True
                    )

                # --- SİTE İÇİ HESAPLAMA ARACI (EN ALTTA) ---
                st.write("")
                st.markdown("---")
                st.subheader(f"🧮 {kod_giris} Yatırım Simülatörü")
                
                col_calc1, col_calc2, col_calc3 = st.columns([1, 1, 1])
                with col_calc1:
                    yatirim_tutar = st.number_input("Yatırılacak Tutar (TL)", min_value=0, value=10000, step=1000)
                with col_calc2:
                    hedef_fiyat = st.number_input("Hedeflediğin Fiyat (Opsiyonel)", min_value=0.0, value=0.0, step=0.1)
                
                # Hesaplama Sonucu
                lot_sayisi = int(yatirim_tutar / son_fiyat)
                kalan_tl = yatirim_tutar - (lot_sayisi * son_fiyat)
                
                with col_calc3:
                    st.metric("Alınabilecek Lot", f"{lot_sayisi} Adet", f"Kalan: {kalan_tl:.2f} TL")
                
                if hedef_fiyat > 0:
                    beklenen_kar = (lot_sayisi * hedef_fiyat) - yatirim_tutar
                    renk_kar = "normal" if beklenen_kar > 0 else "inverse"
                    st.success(f"Eğer fiyat **{hedef_fiyat} TL** olursa kar/zarar durumun: **{beklenen_kar:,.2f} TL** olur.")

            else:
                st.error("Veri bulunamadı. Lütfen hisse kodunu kontrol edin (Örn: SASA).")

# --- DİĞER SAYFALAR (GÖSTERİMLİK) ---
elif sayfa == "🏆 Canlı Piyasa":
    st.title("🏆 Canlı Piyasa (BIST 30)")
    st.info("Bu modül V14.0 kodlarında mevcuttur, demo olarak buradadır.")
elif sayfa == "⚖️ Portföy":
    st.title("⚖️ Akıllı Portföy")
    st.info("Markowitz hesaplayıcı buraya gelecektir.")
