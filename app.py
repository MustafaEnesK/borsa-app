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
st.set_page_config(page_title="BorsApp - AI Terminal", page_icon="📈", layout="wide")

# --- SESSION STATE ---
if 'analiz_aktif' not in st.session_state: st.session_state.analiz_aktif = False
if 'secilen_hisse' not in st.session_state: st.session_state.secilen_hisse = ""

# --- LİSTELER ---
@st.cache_data(ttl=86400)
def listeleri_hazirla():
    # BIST 30 (Sabit Devler)
    bist30 = sorted(["AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "DOAS", "EKGYO", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", "KOZAL", "KRDMD", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK"])
    
    # BIST 100 (Yedek Liste + Scraping Denemesi)
    yedek_bist100 = bist30 + ["AEFES", "AGHOL", "AHGAZ", "AKFGY", "AKSA", "ALGYO", "BERA", "CANTE", "CIMSA", "EGEEN", "ENJSA", "EUPWR", "GESAN", "GWIND", "HALKB", "ISGYO", "IZMDC", "KCAER", "MAVI", "MGROS", "MIATK", "ODAS", "OTKAR", "QUAGR", "REEDR", "SKBNK", "SMRTG", "SOKM", "TAVHL", "TKFEN", "TTKOM", "ULKER", "VAKBN", "VESBE", "YEOTK", "YYLGD", "ZOREN", "ALFAS", "BIOEN", "BOBET", "CWENE", "EBEBK", "EUREN", "GENIL", "KMPUR", "KONYA", "KOPOL", "KOZAA", "KZBGY", "PENTA", "SDTTR", "SNGYO", "SUWEN", "TUKAS", "TURSG"]
    
    try:
        url = "https://tr.wikipedia.org/wiki/BIST_100_endeksine_dahil_hisseler"
        tablolar = pd.read_html(url)
        df = tablolar[0]
        if 'Kod' in df.columns:
            scraped = [str(x).strip().upper() for x in df['Kod'].tolist()]
            return bist30, sorted(list(set(scraped)))
    except: pass
    
    return bist30, sorted(list(set(yedek_bist100)))

BIST_30_LISTESI, BIST_100_LISTESI = listeleri_hazirla()
# BIST 100 içinden BIST 30'u çıkarırsak geriye potansiyel yan tahtalar kalır
GIZLI_CEVHERLER = [x for x in BIST_100_LISTESI if x not in BIST_30_LISTESI]

# --- VERİ ÇEKME ---
@st.cache_data(ttl=600) 
def veri_cek(kod):
    if not kod.endswith(".IS"): kod += ".IS"
    df = yf.download(kod, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

@st.cache_data(ttl=60) # 1 Dakikada bir önbellek yenilenir (Canlı Hissi)
def canli_piyasa_tablosu(hisse_listesi):
    """
    Seçilen listenin anlık fiyat, değişim ve hacim verilerini çeker.
    """
    semboller = [h + ".IS" for h in hisse_listesi]
    # Toplu İndirme (Hız için)
    try:
        data = yf.download(" ".join(semboller), period="2d", interval="1d", group_by='ticker', auto_adjust=True, progress=False)
    except: return pd.DataFrame()

    tablo_verisi = []
    
    for h in hisse_listesi:
        try:
            df = data[h + ".IS"]
            if df.empty: continue
            
            # Son gün ve önceki gün kapanışı
            # Eğer piyasa açıksa son satır anlık fiyat, bir önceki dünün kapanışıdır.
            son_fiyat = df['Close'].iloc[-1]
            onceki_kapanis = df['Close'].iloc[-2]
            degisim_yuzde = ((son_fiyat - onceki_kapanis) / onceki_kapanis) * 100
            hacim = df['Volume'].iloc[-1]
            
            durum = "NÖTR ⚪"
            if degisim_yuzde > 3: durum = "GÜÇLÜ ALICI 🟢🟢"
            elif degisim_yuzde > 0: durum = "POZİTİF 🟢"
            elif degisim_yuzde < -3: durum = "GÜÇLÜ SATICI 🔴🔴"
            elif degisim_yuzde < 0: durum = "NEGATİF 🔴"
            
            tablo_verisi.append({
                "Kod": h,
                "Fiyat": son_fiyat,
                "Değişim %": degisim_yuzde,
                "Hacim": hacim,
                "Durum": durum
            })
        except: continue
        
    return pd.DataFrame(tablo_verisi)

# --- AI MODELLERİ ---
def simulasyon_hesapla(df, gun_sayisi):
    try:
        df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        m = Prophet(daily_seasonality=True)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=gun_sayisi)
        forecast = m.predict(future)
        return forecast['yhat'].iloc[-1]
    except: return None

def xgboost_sinyal(df):
    data = df.copy()
    try:
        data['RSI'] = ta.rsi(data['Close'], 14)
        data['SMA'] = ta.sma(data['Close'], 50)
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        data.dropna(inplace=True)
        if len(data) < 50: return None, None, 0
        X = data[['RSI', 'SMA', 'Volume']]
        y = data['Target']
        X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, shuffle=False)
        model = XGBClassifier(eval_metric='logloss')
        model.fit(X_train, y_train)
        last = X.iloc[[-1]]
        return model.predict(last)[0], model.predict_proba(last)[0], accuracy_score(y_test, model.predict(X_test))
    except: return None, None, 0

def markowitz(hisseler, butce):
    if len(hisseler)<2: return None, "Min 2 hisse seç."
    data = pd.DataFrame()
    for h in hisseler:
        d = veri_cek(h)
        if not d.empty: data[h] = d.set_index('Date')['Close']
    data.dropna(inplace=True)
    if len(data)<50: return None, "Veri yetersiz."
    ret = data.pct_change().dropna()
    mu, sigma = ret.mean().values, ret.cov().values
    w = cp.Variable(len(data.columns))
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, sigma)), [cp.sum(w)==1, w>=0])
    try:
        prob.solve()
        return dict(zip(data.columns, np.round(w.value,3))), None
    except: return None, "Hata."

# --- ARAYÜZ ---
with st.sidebar:
    st.markdown("""
    <div style="background-color:#0E1117; padding:20px; border-radius:15px; border:1px solid #262730; text-align:center;">
        <h1 style="color:#00CC96; margin:0; font-size: 32px; font-weight: 800;">BorsApp 🚀</h1>
        <p style="color:#FAFAFA; font-size:14px; margin-top:5px; font-weight: 600;">AI Trading & Finans Simülatörü</p>
        <hr style="margin:15px 0; border-color:#333;">
        <p style="color:#9CA0A6; font-size:12px; margin:0;">Geliştirici: Mustafa Enes KORKMAZOĞLU</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    col_s1, col_s2 = st.columns(2)
    with col_s1: st.link_button("👔 LinkedIn", "https://www.linkedin.com/in/mustafa-enes-korkmazoglu", use_container_width=True)
    with col_s2: st.link_button("📸 Instagram", "https://www.instagram.com/mustafaenesk_", use_container_width=True)
    st.divider()

    st.header("📲 Menü")
    # YENİ MENÜ YAPISI: BIST 30 ve 100 Eklendi
    sayfa = st.radio("Modül Seçiniz:", 
                     ["🔎 Serbest Analiz", 
                      "🏆 BIST 30 Canlı Takip", 
                      "💯 BIST 100 Canlı Takip", 
                      "🕵️‍♂️ Keşif Taraması", 
                      "⚖️ Akıllı Portföy"])
    
    st.divider()
    
    with st.expander("💰 Hızlı Yatırım Hesapla", expanded=False):
        sim_hisse = st.selectbox("Hisse", ["THYAO", "ASELS"] + BIST_30_LISTESI)
        sim_tutar = st.number_input("Tutar (TL)", 1000, 1000000, 10000, step=1000)
        vade_etiket = st.select_slider("Vade", options=["15 Gün", "1 Ay", "3 Ay"])
        if st.button("Hesapla 🧮"):
            gun_map = {"15 Gün": 15, "1 Ay": 30, "3 Ay": 90}
            df_sim = veri_cek(sim_hisse)
            if not df_sim.empty:
                fiyat = df_sim['Close'].iloc[-1]
                tahmin = simulasyon_hesapla(df_sim, gun_map[vade_etiket])
                lot = int(sim_tutar/fiyat)
                gelecek = lot * tahmin
                fark = gelecek - sim_tutar
                renk = "normal" if fark >= 0 else "inverse"
                st.metric("TAHMİNİ KAR/ZARAR", f"{gelecek:,.0f} TL", f"{fark:,.0f} TL", delta_color=renk)

# --- SAYFA 1: SERBEST ANALİZ ---
if sayfa == "🔎 Serbest Analiz":
    st.title("🔎 Serbest Hisse Analizi")
    st.info("Listelerde olmayan herhangi bir hisse kodunu (Örn: TABGD, EBEBK) girerek analiz edebilirsin.")
    col1, col2 = st.columns([3,1])
    with col1: kod = st.text_input("Hisse Kodu", "THYAO").upper()
    with col2: 
        st.write("")
        st.write("")
        btn = st.button("Analiz Et 🚀", type="primary", use_container_width=True)
    
    if btn or kod:
        with st.spinner("Veriler getiriliyor..."):
            df = veri_cek(kod)
            if not df.empty:
                son = df['Close'].iloc[-1]
                deg = ((son - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100
                st.metric(f"{kod} Fiyatı", f"{son:.2f} ₺", f"%{deg:.2f}")
                
                # Grafik ve AI
                tab1, tab2 = st.tabs(["Grafik", "Yapay Zeka"])
                with tab1:
                    fig = go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                    fig.add_trace(go.Scatter(x=df['Date'], y=ta.sma(df['Close'], 50), line=dict(color='orange'), name='SMA50'))
                    st.plotly_chart(fig, use_container_width=True)
                with tab2:
                    pred, prob, acc = xgboost_sinyal(df)
                    if pred is not None:
                        renk = "green" if pred==1 else "red"
                        yon = "YÜKSELİŞ" if pred==1 else "DÜŞÜŞ"
                        st.markdown(f"### Sinyal: <span style='color:{renk}'>{yon}</span>", unsafe_allow_html=True)
                        st.write(f"Güven: %{max(prob)*100:.1f}")

# --- SAYFA 2: BIST 30 CANLI ---
elif sayfa == "🏆 BIST 30 Canlı Takip":
    st.title("🏆 BIST 30 Canlı Takip Ekranı")
    st.caption("Veriler 15dk gecikmeli olabilir. Sayfayı yenileyerek (veya butona basarak) güncelleyebilirsiniz.")
    
    if st.button("🔄 Verileri Şimdi Güncelle"):
        st.cache_data.clear()
    
    with st.spinner("BIST 30 Hisseleri Çekiliyor..."):
        df_canli = canli_piyasa_tablosu(BIST_30_LISTESI)
        
        if not df_canli.empty:
            # Sıralama: En çok artanlar en üstte
            df_canli = df_canli.sort_values(by="Değişim %", ascending=False)
            
            # Renklendirme
            def renkli_degisim(val):
                color = '#00FF00' if val > 0 else '#FF4B4B' if val < 0 else 'white'
                return f'color: {color}; font-weight: bold'
            
            st.dataframe(
                df_canli,
                column_config={
                    "Kod": st.column_config.TextColumn("Sembol"),
                    "Fiyat": st.column_config.NumberColumn("Son Fiyat", format="%.2f ₺"),
                    "Değişim %": st.column_config.NumberColumn("Günlük Fark", format="%.2f%%"),
                    "Hacim": st.column_config.NumberColumn("Hacim", format="%d"),
                    "Durum": st.column_config.TextColumn("Piyasa Yönü")
                },
                hide_index=True,
                use_container_width=True,
                height=800
            )
            # Stil uygulama (Streamlit dataframe içinde kısıtlıdır, görselleştirme config ile yapıldı)
        else:
            st.error("Piyasa verilerine ulaşılamadı.")

# --- SAYFA 3: BIST 100 CANLI ---
elif sayfa == "💯 BIST 100 Canlı Takip":
    st.title("💯 BIST 100 Geneli - Canlı Takip")
    st.info("BIST 100 Endeksindeki tüm hisselerin anlık performans tablosu.")
    
    if st.button("🔄 Listeyi Yenile"):
        st.cache_data.clear()
        
    with st.spinner("100 Hisse Taranıyor (Bu işlem 3-4 saniye sürebilir)..."):
        df_canli = canli_piyasa_tablosu(BIST_100_LISTESI)
        
        if not df_canli.empty:
            df_canli = df_canli.sort_values(by="Değişim %", ascending=False)
            
            st.dataframe(
                df_canli,
                column_config={
                    "Kod": st.column_config.TextColumn("Hisse"),
                    "Fiyat": st.column_config.NumberColumn("Fiyat", format="%.2f ₺"),
                    "Değişim %": st.column_config.NumberColumn("Günlük %", format="%.2f%%"),
                    "Durum": st.column_config.TextColumn("Sinyal")
                },
                hide_index=True,
                use_container_width=True,
                height=1000 # Daha uzun liste
            )
        else:
            st.error("Veri alınamadı.")

# --- SAYFA 4: KEŞİF ---
elif sayfa == "🕵️‍♂️ Keşif Taraması":
    st.title("🕵️‍♂️ Fırsat Avcısı")
    st.info("BIST 30 Dışındaki Potansiyel Hisseler (Yan Tahtalar)")
    if st.button("Fırsatları Tara 🚀", type="primary"):
        with st.spinner("Taranıyor..."):
            df_tablo = detayli_tarama_yap(GIZLI_CEVHERLER)
            if not df_tablo.empty:
                df_tablo = df_tablo.sort_values(by="30 Günlük", ascending=False)
                st.dataframe(df_tablo, use_container_width=True, height=800)

# --- SAYFA 5: PORTFÖY ---
elif sayfa == "⚖️ Akıllı Portföy":
    st.title("⚖️ Portföy Sihirbazı")
    hisseler = st.multiselect("Hisselerini Seç", BIST_100_LISTESI, default=["THYAO", "ASELS", "GARAN"])
    butce = st.number_input("Bütçe (TL)", 1000, 1000000, 50000)
    if st.button("Dağılımı Hesapla"):
        agirlik, err = markowitz(hisseler, butce)
        if err: st.error(err)
        else:
            df_p = pd.DataFrame(list(agirlik.items()), columns=['Hisse','Oran'])
            df_p = df_p[df_p['Oran']>0.01]
            c1,c2 = st.columns(2)
            c1.plotly_chart(px.pie(df_p, values='Oran', names='Hisse'), use_container_width=True)
            c2.write("### Alınacaklar")
            for i, r in df_p.iterrows():
                st.write(f"• **{r['Hisse']}**: {butce*r['Oran']:,.0f} TL")
