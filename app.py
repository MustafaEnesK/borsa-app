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

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="NEÜ Finansal Zeka", page_icon="🧠", layout="wide")

# --- SESSION STATE ---
if 'analiz_aktif' not in st.session_state: st.session_state.analiz_aktif = False
if 'secilen_hisse' not in st.session_state: st.session_state.secilen_hisse = ""

# --- FONKSİYONLAR ---

@st.cache_data(ttl=86400)
def bist_listesini_getir():
    yedek = ["AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "DOAS", "EKGYO", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", "KOZAL", "KRDMD", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK"]
    try:
        url = "https://tr.wikipedia.org/wiki/BIST_100_endeksine_dahil_hisseler"
        tablolar = pd.read_html(url)
        df = tablolar[0]
        if 'Kod' in df.columns:
            return sorted([str(x).strip().upper() for x in df['Kod'].tolist()])
        return sorted(yedek)
    except: return sorted(yedek)

TUM_HISSELER = bist_listesini_getir()

@st.cache_data(ttl=600) 
def veri_cek(kod):
    if not kod.endswith(".IS"): kod += ".IS"
    df = yf.download(kod, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

@st.cache_data(ttl=3600)
def temel_analiz_getir(kod):
    if not kod.endswith(".IS"): kod += ".IS"
    try:
        info = yf.Ticker(kod).info
        return {"F/K": info.get("trailingPE", None), "PD/DD": info.get("priceToBook", None), "Beta": info.get("beta", None), "Temettü": info.get("dividendYield", None)}
    except: return None

# --- GELİŞMİŞ MODELLER ---

def xgboost_sinyal(df):
    """
    XGBoost kullanarak yarın hissenin yükselip yükselmeyeceğini tahmin eder.
    """
    data = df.copy()
    # Feature Engineering (İndikatörler)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data['SMA_Diff'] = data['Close'] - ta.sma(data['Close'], length=50)
    data['Return'] = data['Close'].pct_change()
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int) # Yarın artarsa 1, düşerse 0
    
    data.dropna(inplace=True)
    
    features = ['RSI', 'SMA_Diff', 'Return', 'Volume']
    X = data[features]
    y = data['Target']
    
    # Son gün hariç eğitim, son gün tahmin
    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, shuffle=False)
    
    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)
    
    # Başarı oranı (Test seti)
    acc = accuracy_score(y_test, model.predict(X_test))
    
    # Gelecek Tahmini (Bugünün verisiyle yarını tahmin et)
    last_row = X.iloc[[-1]]
    prediction = model.predict(last_row)[0]
    prob = model.predict_proba(last_row)[0]
    
    return prediction, prob, acc

def statsmodels_analiz(df):
    """
    Zaman serisi ayrıştırması (Trend, Mevsimsellik)
    """
    df_ts = df.set_index('Date')['Close'].asfreq('B') # İş günleri
    df_ts = df_ts.fillna(method='ffill')
    res = sm.tsa.seasonal_decompose(df_ts, model='additive', period=20) # Aylık periyot
    return res

def markowitz_optimize(hisseler, butce):
    if len(hisseler) < 2: return None, "En az 2 hisse seç."
    
    # Veri Hazırlığı
    data = pd.DataFrame()
    for h in hisseler:
        try:
            df = veri_cek(h)
            if not df.empty: data[h] = df.set_index('Date')['Close']
        except: continue
    
    data.dropna(inplace=True)
    if data.empty: return None, "Veri yetersiz."
    
    # Markowitz (CVXPY)
    mu = data.pct_change().mean().values
    sigma = data.pct_change().cov().values
    n = len(data.columns)
    
    w = cp.Variable(n)
    risk = cp.quad_form(w, sigma)
    # Amaç: Riski minimize et, toplam ağırlık 1 olsun, short yok
    prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1, w >= 0])
    prob.solve()
    
    return dict(zip(data.columns, np.round(w.value, 3))), None

# --- YAN MENÜ (DÜZENLİ) ---
with st.sidebar:
    # 1. KİMLİK KARTI
    st.markdown("""
    <div style="background-color:#0E1117; padding:15px; border-radius:10px; border:1px solid #262730; text-align:center;">
        <h3 style="color:#FAFAFA; margin:0;">Mustafa Enes KORKMAZOĞLU</h3>
        <p style="color:#9CA0A6; font-size:12px; margin:5px 0;">NEÜ İktisat 3. Sınıf</p>
        <div style="background-color:#262730; color:#00CC96; padding:5px; border-radius:5px; font-weight:bold; font-size:12px;">
            V6.0 ULTIMATE
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    col_social1, col_social2 = st.columns(2)
    with col_social1: st.link_button("👔 LinkedIn", "https://www.linkedin.com/in/mustafa-enes-korkmazoglu", use_container_width=True)
    with col_social2: st.link_button("📸 Instagram", "https://www.instagram.com/mustafaenesk_", use_container_width=True)
    st.divider()

    # 2. MODÜL SEÇİMİ (RADYO BUTON İLE DÜZENLİ)
    st.header("📲 Modül Seçimi")
    modul = st.radio(
        "Çalışma Alanı:",
        ["📈 Hisse Analiz (Detaylı)", "⚖️ Portföy Optimize (CVXPY)", "💎 Keşif Listesi"],
        captions=["AI, Ekonometri ve Teknik", "Markowitz Modeli", "Potansiyelli Hisseler"]
    )
    st.divider()

    # 3. DİNAMİK AYARLAR (Modüle göre değişir)
    if modul == "📈 Hisse Analiz (Detaylı)":
        st.subheader("Hisse Ayarları")
        liste_tipi = st.selectbox("Liste Kaynağı", ["BIST 30", "BIST 100", "Manuel"])
        
        if liste_tipi == "Manuel": secilen = st.text_input("Kod", "THYAO").upper()
        elif liste_tipi == "BIST 30": secilen = st.selectbox("Hisse", sorted(["AKBNK", "THYAO", "ASELS", "GARAN", "EREGL", "TUPRS", "KCHOL"])) # Örnek
        else: secilen = st.selectbox("Hisse", TUM_HISSELER)
        
        btn_calistir = st.button("Analiz Et 🚀", type="primary", use_container_width=True)

    elif modul == "⚖️ Portföy Optimize (CVXPY)":
        st.subheader("Portföy Ayarları")
        portfoy_hisseleri = st.multiselect("Hisseler (Min 2)", TUM_HISSELER, default=["THYAO", "ASELS", "GARAN"])
        butce = st.number_input("Bütçe (TL)", 1000, 1000000, 50000)
        btn_calistir = st.button("Optimize Et ⚖️", type="primary", use_container_width=True)

    elif modul == "💎 Keşif Listesi":
        st.info("Büyüme potansiyeli yüksek ve temettü hisseleri.")
        kesif_hisseleri = ["ALFAS", "EGEEN", "FROTO", "GESAN", "KONTR", "MIATK", "YEOTK", "SMRTG"]
        secilen = st.selectbox("Hisse Seç", kesif_hisseleri)
        btn_calistir = st.button("İncele 🔍", type="primary", use_container_width=True)


# --- ANA EKRAN ---
st.title("🧠 NEÜ Finansal Zeka Platformu")

# --- MODÜL 1 & 3: HİSSE ANALİZİ ---
if modul in ["📈 Hisse Analiz (Detaylı)", "💎 Keşif Listesi"]:
    if btn_calistir:
        st.session_state.analiz_aktif = True
        st.session_state.secilen_hisse = secilen
        st.cache_data.clear()

    if st.session_state.analiz_aktif:
        kod = st.session_state.secilen_hisse
        
        with st.spinner("Veriler çekiliyor, XGBoost ve Ekonometrik modeller çalışıyor..."):
            df = veri_cek(kod)
            temel = temel_analiz_getir(kod)
            
            if df.empty:
                st.error("Veri bulunamadı.")
            else:
                # ÜST METRİKLER
                son = df['Close'].iloc[-1]
                degisim = ((son - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Fiyat", f"{son:.2f} ₺", f"%{degisim:.2f}")
                if temel:
                    c2.metric("F/K", temel['F/K'] if temel['F/K'] else "-")
                    c3.metric("PD/DD", temel['PD/DD'] if temel['PD/DD'] else "-")
                    c4.metric("Beta (Risk)", f"{temel['Beta']:.2f}" if temel['Beta'] else "-")

                # --- SEKMELİ YAPI (TIDY UI) ---
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Teknik & Grafik", "🤖 Yapay Zeka (XGBoost)", "📉 Ekonometri (Stats)", "📰 Haberler"])
                
                with tab1: # TEKNİK
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'))
                    fig.add_trace(go.Scatter(x=df['Date'], y=ta.sma(df['Close'], 50), line=dict(color='orange'), name='SMA 50'))
                    fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2: # AI & XGBOOST
                    col_ai1, col_ai2 = st.columns([1,2])
                    
                    pred, prob, acc = xgboost_sinyal(df)
                    guven = max(prob) * 100
                    yon = "YÜKSELİŞ 🚀" if pred == 1 else "DÜŞÜŞ 🔻"
                    renk = "green" if pred == 1 else "red"
                    
                    with col_ai1:
                        st.subheader("XGBoost Sinyali")
                        st.markdown(f"<h2 style='color:{renk};'>{yon}</h2>", unsafe_allow_html=True)
                        st.write(f"**Güven:** %{guven:.2f}")
                        st.write(f"**Model Başarısı:** %{acc*100:.1f}")
                        st.caption("*Gradient Boosting algoritması teknik indikatörleri yorumlayarak yarınki yönü tahmin eder.")
                    
                    with col_ai2:
                        st.subheader("Prophet (Uzun Vade)")
                        m = Prophet()
                        m.fit(df.rename(columns={'Date':'ds', 'Close':'y'}))
                        future = m.make_future_dataframe(periods=30)
                        fcast = m.predict(future)
                        fig_p = go.Figure()
                        fig_p.add_trace(go.Scatter(x=fcast['ds'], y=fcast['yhat'], line=dict(color='cyan'), name='Tahmin'))
                        fig_p.add_trace(go.Scatter(x=df['Date'], y=df['Close'], line=dict(color='white', width=1), name='Gerçek'))
                        fig_p.update_layout(height=350, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
                        st.plotly_chart(fig_p, use_container_width=True)

                with tab3: # EKONOMETRİ (STATSMODELS)
                    st.subheader("Zaman Serisi Ayrıştırması (Decomposition)")
                    st.info("Hissenin fiyatını Trend, Mevsimsellik ve Rassal (Gürültü) bileşenlerine ayırır.")
                    res = statsmodels_analiz(df)
                    
                    c_s1, c_s2 = st.columns(2)
                    with c_s1:
                        st.write("**Trend (Ana Yön)**")
                        st.line_chart(res.trend, color="#FFA500")
                    with c_s2:
                        st.write("**Mevsimsellik (Döngü)**")
                        st.line_chart(res.seasonal, color="#00CC96")
                    
                    st.write("**Resid (Rassal/Gürültü)**")
                    st.line_chart(res.resid, color="#FF4B4B")

                with tab4: # HABERLER
                    news = feedparser.parse(f"https://news.google.com/rss/search?q={kod}+hisse&hl=tr&gl=TR&ceid=TR:tr").entries[:5]
                    if news:
                        for n in news: st.info(f"[{n.title}]({n.link})")
                    else: st.write("Haber bulunamadı.")

# --- MODÜL 2: PORTFÖY OPTİMİZASYONU ---
elif modul == "⚖️ Portföy Optimize (CVXPY)":
    if btn_calistir:
        with st.spinner("Markowitz (Mean-Variance) Modeli Çalışıyor..."):
            agirliklar, hata = markowitz_optimize(portfoy_hisseleri, butce)
            
            if hata: st.error(hata)
            else:
                st.success("✅ Optimum Dağılım Hesaplandı!")
                c_p1, c_p2 = st.columns([1,1])
                
                # Sadece %1 üzeri olanları al
                df_pie = pd.DataFrame(list(agirliklar.items()), columns=['Hisse', 'Oran'])
                df_pie = df_pie[df_pie['Oran'] > 0.01]
                
                with c_p1:
                    fig_pie = px.pie(df_pie, values='Oran', names='Hisse', title='İdeal Cüzdan (Risk Minimize)', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with c_p2:
                    st.subheader("💰 Alım Emri")
                    st.write(f"**{butce:,.0f} TL** bütçe ile önerilen dağılım:")
                    for idx, row in df_pie.iterrows():
                        tutar = butce * row['Oran']
                        st.write(f"• **{row['Hisse']}**: %{row['Oran']*100:.1f} -> **{tutar:,.0f} TL**")
else:
    st.info("👈 Sol menüden modül seçerek başlayın.")
