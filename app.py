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
st.set_page_config(page_title="BorsApp - AI Trading", page_icon="📈", layout="wide")

# --- SESSION STATE ---
if 'analiz_aktif' not in st.session_state: st.session_state.analiz_aktif = False
if 'secilen_hisse' not in st.session_state: st.session_state.secilen_hisse = ""

# --- CANLI VERİ KAZIMA ---
@st.cache_data(ttl=43200) 
def tum_hisseleri_guncelle():
    yedek_liste = ["ALFAS", "ASTOR", "BIOEN", "BOBET", "BRSAN", "BURCE", "CANTE", "CEMTS", "CVKMD", "CWENE", "DAPGM", "EGEEN", "ENJSA", "EUPWR", "FROTO", "GENIL", "GESAN", "GWIND", "HKTM", "HUNER", "INVEO", "ISMEN", "IZMDC", "JANTS", "KCAER", "KLKIM", "KMPUR", "KNFRT", "KONTR", "MIATK", "MOBTL", "NATEN", "ODAS", "OTKAR", "OYLUM", "OZSUB", "PENTA", "QUAGR", "REEDR", "RUBNS", "SDTTR", "SMRTG", "SNGYO", "SOKM", "SUWEN", "TATGD", "TKFEN", "TTRAK", "VBTYZ", "YEOTK", "YYLGD", "ZOREN"]
    try:
        url = "https://tr.wikipedia.org/wiki/Borsa_%C4%B0stanbul%27da_i%C5%9Flem_g%C3%B6ren_%C5%9Firketler_listesi"
        tablolar = pd.read_html(url)
        tum_kodlar = []
        for tablo in tablolar:
            if 'Kod' in tablo.columns:
                kodlar = tablo['Kod'].tolist()
                tum_kodlar.extend([str(k).strip().upper() for k in kodlar])
        temiz_liste = sorted(list(set(tum_kodlar)))
        return temiz_liste if len(temiz_liste) > 50 else yedek_liste
    except: return yedek_liste

TUM_HISSELER_CANLI = tum_hisseleri_guncelle()
BIST_30_SABIT = ["AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "DOAS", "EKGYO", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", "KOZAL", "KRDMD", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK"]
GIZLI_CEVHERLER_DINAMIK = [h for h in TUM_HISSELER_CANLI if h not in BIST_30_SABIT]

# --- VERİ ÇEKME ---
@st.cache_data(ttl=600) 
def veri_cek(kod):
    kod = kod.upper().strip()
    if not kod.endswith(".IS"): kod += ".IS"
    df = yf.download(kod, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

@st.cache_data(ttl=3600)
def detayli_tarama_yap(hisse_listesi):
    semboller = [h + ".IS" for h in hisse_listesi]
    tarama_limiti = hisse_listesi[:150] 
    semboller_limit = [h + ".IS" for h in tarama_limiti]
    try:
        data = yf.download(" ".join(semboller_limit), period="6mo", interval="1d", group_by='ticker', auto_adjust=True, progress=False)
    except: return pd.DataFrame()

    rapor = []
    for hisse in tarama_limiti:
        try:
            df = data[hisse + ".IS"].copy()
            if df.empty or len(df) < 95: continue 
            son_fiyat = df['Close'].iloc[-1]
            fiyat_30g = df['Close'].iloc[-21]
            getiri_30 = ((son_fiyat - fiyat_30g) / fiyat_30g)
            rsi = ta.rsi(df['Close'], 14).iloc[-1]
            sma50 = ta.sma(df['Close'], 50).iloc[-1]
            trend = "Yükseliş ↗️" if son_fiyat > sma50 else "Düşüş ↘️"
            rapor.append({
                "Hisse": hisse, "Fiyat": son_fiyat, "30 Günlük": getiri_30, 
                "RSI": rsi, "Trend": trend
            })
        except: continue
    return pd.DataFrame(rapor)

# --- AI & SİMÜLASYON ---
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
        <p style="color:#9CA0A6; font-size:12px; margin:0;">NEÜ İktisat 3. Sınıf</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    col_s1, col_s2 = st.columns(2)
    with col_s1: st.link_button("👔 LinkedIn", "https://www.linkedin.com/in/mustafa-enes-korkmazoglu", use_container_width=True)
    with col_s2: st.link_button("📸 Instagram", "https://www.instagram.com/mustafaenesk_", use_container_width=True)
    st.divider()

    st.header("📲 Kontrol Paneli")
    # YENİ MENÜ EKLENDİ: "🔎 Manuel/Serbest Analiz"
    sayfa = st.radio("Modül Seçiniz:", ["🔎 Manuel/Serbest Analiz", "🕵️‍♂️ Canlı Keşif Taraması", "📈 Liste Bazlı AI Analizi", "⚖️ Akıllı Portföy"])
    
    st.divider()
    
    with st.expander("💰 AI Getiri Simülatörü", expanded=True):
        sim_hisse = st.selectbox("Hisse Seç", ["THYAO", "ASELS", "GARAN"] + TUM_HISSELER_CANLI[:50])
        sim_tutar = st.number_input("Yatırım Tutarı (TL)", 1000, 1000000, 10000, step=1000)
        vade_etiket = st.select_slider("Vade Seç", options=["15 Gün", "1 Ay", "3 Ay", "6 Ay"])
        gun_map = {"15 Gün": 15, "1 Ay": 30, "3 Ay": 90, "6 Ay": 180}
        
        if st.button("Getiriyi Hesapla 🧮", type="primary"):
            with st.spinner("AI Hesaplıyor..."):
                df_sim = veri_cek(sim_hisse)
                if not df_sim.empty:
                    anlik_fiyat = df_sim['Close'].iloc[-1]
                    tahmini_fiyat = simulasyon_hesapla(df_sim, gun_map[vade_etiket])
                    if tahmini_fiyat:
                        lot = int(sim_tutar / anlik_fiyat)
                        gelecek_tutar = lot * tahmini_fiyat
                        fark = gelecek_tutar - sim_tutar
                        renk = "normal" if fark >= 0 else "inverse"
                        st.metric("TAHMİNİ KAR/ZARAR", f"{gelecek_tutar:,.0f} TL", f"{fark:,.0f} TL", delta_color=renk)
                        st.caption(f"{vade_etiket} sonraki AI tahmini: {tahmini_fiyat:.2f} TL")
                    else: st.error("Hesaplanamadı.")

# --- SAYFA 4: MANUEL/SERBEST ANALİZ (YENİ MODÜL) ---
if sayfa == "🔎 Manuel/Serbest Analiz":
    st.title("🔎 Serbest Hisse Arama Modülü")
    st.info("Listelerde bulamadığınız veya özel olarak incelemek istediğiniz hisse kodunu aşağıya yazın.")
    
    col_input1, col_input2 = st.columns([3, 1])
    with col_input1:
        girilen_kod = st.text_input("BIST Kodu Giriniz (Örn: KONTR, EBEBK, TABGD)", "THYAO").upper()
    with col_input2:
        st.write("")
        st.write("")
        btn_ara = st.button("Hisse Getir 🔎", type="primary", use_container_width=True)
    
    if btn_ara or girilen_kod:
        with st.spinner(f"{girilen_kod} verileri getiriliyor..."):
            df = veri_cek(girilen_kod)
            if not df.empty:
                st.success(f"✅ {girilen_kod} Verisi Bulundu!")
                son = df['Close'].iloc[-1]
                deg = ((son - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100
                
                # Temel Verileri Çekmeye Çalış
                try:
                    info = yf.Ticker(girilen_kod+".IS").info
                    fk = info.get('trailingPE', '-')
                    pd_dd = info.get('priceToBook', '-')
                except: fk, pd_dd = "-", "-"

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Fiyat", f"{son:.2f} ₺", f"%{deg:.2f}")
                c2.metric("RSI", f"{ta.rsi(df['Close'], 14).iloc[-1]:.1f}")
                c3.metric("F/K", fk if fk!='-' else '-')
                c4.metric("PD/DD", pd_dd if pd_dd!='-' else '-')

                tab1, tab2 = st.tabs(["📊 Grafik & Teknik", "🧠 Yapay Zeka"])
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
                    
                    m = Prophet()
                    m.fit(df.rename(columns={'Date':'ds', 'Close':'y'}))
                    future = m.make_future_dataframe(periods=30)
                    fcast = m.predict(future)
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(x=fcast['ds'], y=fcast['yhat'], line=dict(color='cyan'), name='Tahmin'))
                    fig_p.add_trace(go.Scatter(x=df['Date'], y=df['Close'], line=dict(color='white'), name='Gerçek'))
                    st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.error("Hisse bulunamadı veya veri çekilemedi. Kodun doğru olduğundan emin olun.")

# --- SAYFA 1: OTOMATİK KEŞİF ---
elif sayfa == "🕵️‍♂️ Canlı Keşif Taraması":
    st.title("🕵️‍♂️ Canlı Piyasa Taraması")
    st.info(f"Sistemdeki **{len(TUM_HISSELER_CANLI)}** hisse arasından fırsatlar taranıyor.")
    if st.button("Fırsatları Tara 🚀", type="primary"):
        with st.spinner("Piyasa taranıyor..."):
            df_tablo = detayli_tarama_yap(GIZLI_CEVHERLER_DINAMIK)
            if not df_tablo.empty:
                df_tablo = df_tablo.sort_values(by="30 Günlük", ascending=False)
                st.dataframe(
                    df_tablo,
                    column_config={
                        "Hisse": st.column_config.TextColumn("Kod"),
                        "Fiyat": st.column_config.NumberColumn("Fiyat", format="%.2f ₺"),
                        "30 Günlük": st.column_config.ProgressColumn("1 Ay Getiri", format="%.2f%%", min_value=-0.5, max_value=0.5),
                        "RSI": st.column_config.NumberColumn("RSI", help="30 altı fırsat"),
                        "Trend": st.column_config.TextColumn("Yön")
                    }, hide_index=True, use_container_width=True, height=800
                )
            else: st.error("Veri alınamadı.")

# --- SAYFA 2: LİSTE BAZLI ANALİZ ---
elif sayfa == "📈 Liste Bazlı AI Analizi":
    st.title("📈 Listeden Hızlı Analiz")
    secilen = st.selectbox("Analiz Edilecek Hisse", TUM_HISSELER_CANLI)
    if st.button("Analizi Başlat"):
        with st.spinner("AI Modelleri Çalışıyor..."):
            df = veri_cek(secilen)
            if not df.empty:
                son = df['Close'].iloc[-1]
                deg = ((son - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100
                st.metric("Fiyat", f"{son:.2f} ₺", f"%{deg:.2f}")
                
                fig = go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                fig.add_trace(go.Scatter(x=df['Date'], y=ta.sma(df['Close'], 50), line=dict(color='orange'), name='SMA50'))
                st.plotly_chart(fig, use_container_width=True)

# --- SAYFA 3: PORTFÖY ---
elif sayfa == "⚖️ Akıllı Portföy":
    st.title("⚖️ Portföy Sihirbazı")
    hisseler = st.multiselect("Portföy Oluştur", TUM_HISSELER_CANLI, default=["THYAO", "ASELS", "GARAN"])
    butce = st.number_input("Bütçe (TL)", 1000, 1000000, 50000)
    if st.button("Optimize Et"):
        agirlik, err = markowitz(hisseler, butce)
        if err: st.error(err)
        else:
            df_p = pd.DataFrame(list(agirlik.items()), columns=['Hisse','Oran'])
            df_p = df_p[df_p['Oran']>0.01]
            c1,c2 = st.columns(2)
            c1.plotly_chart(px.pie(df_p, values='Oran', names='Hisse'), use_container_width=True)
            c2.write("### Alım Emri")
            for i, r in df_p.iterrows():
                st.write(f"• **{r['Hisse']}**: {butce*r['Oran']:,.0f} TL")
