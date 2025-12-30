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
st.set_page_config(page_title="NEÜ Finansal Zeka", page_icon="🧠", layout="wide")

# --- SESSION STATE ---
if 'analiz_aktif' not in st.session_state: st.session_state.analiz_aktif = False
if 'secilen_hisse' not in st.session_state: st.session_state.secilen_hisse = ""

# --- LİSTE ÇEKİCİ ---
@st.cache_data(ttl=86400)
def bist_100_getir():
    yedek = ["AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "DOAS", "EKGYO", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", "KOZAL", "KRDMD", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK"]
    try:
        url = "https://tr.wikipedia.org/wiki/BIST_100_endeksine_dahil_hisseler"
        tablolar = pd.read_html(url)
        df = tablolar[0]
        if 'Kod' in df.columns:
            return sorted([str(x).strip().upper() for x in df['Kod'].tolist()])
        return sorted(yedek)
    except: return sorted(yedek)

TUM_HISSELER = bist_100_getir()

# --- ÖZEL LİSTE: GİZLİ CEVHERLER (YAN TAHTALAR) ---
# Bu liste BIST 30 dışındaki, yüksek potansiyelli, teknoloji ve enerji ağırlıklı hisselerdir.
GIZLI_CEVHERLER = [
    "ALFAS", "ASTOR", "BIOEN", "BOBET", "BRSAN", "BURCE", "CANTE", "CEMTS", 
    "CVKMD", "CWENE", "DAPGM", "EGEEN", "ENJSA", "EUPWR", "FROTO", "GENIL", 
    "GESAN", "GLYHO", "GWIND", "HKTM", "HUNER", "INVEO", "ISMEN", "IZMDC", 
    "JANTS", "KCAER", "KLKIM", "KMPUR", "KNFRT", "KONTR", "MIATK", "MOBTL", 
    "NATEN", "ODAS", "OTKAR", "OYLUM", "OZSUB", "PENTA", "QUAGR", "REEDR", 
    "RUBNS", "SDTTR", "SMRTG", "SNGYO", "SOKM", "SUWEN", "TATGD", "TKFEN", 
    "TTRAK", "VBTYZ", "YEOTK", "YYLGD", "ZOREN"
]

# --- VERİ ÇEKME FONKSİYONLARI ---
@st.cache_data(ttl=600) 
def veri_cek(kod):
    if not kod.endswith(".IS"): kod += ".IS"
    df = yf.download(kod, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

@st.cache_data(ttl=3600)
def detayli_tarama_yap(hisse_listesi):
    """
    Hisseleri tarar ve 30-60-90 günlük getirilerini hesaplar.
    """
    semboller = [h + ".IS" for h in hisse_listesi]
    
    # Batch Download (Hızlı - 6 aylık veri çekiyoruz ki 90 günü hesaplayabilelim)
    try:
        data = yf.download(" ".join(semboller), period="6mo", interval="1d", group_by='ticker', auto_adjust=True, progress=False)
    except:
        return pd.DataFrame()

    rapor = []
    
    for hisse in hisse_listesi:
        try:
            df = data[hisse + ".IS"].copy()
            if df.empty or len(df) < 95: continue # En az 95 gün veri lazım
            
            # Anlık Fiyat
            son_fiyat = df['Close'].iloc[-1]
            
            # Getiri Hesaplamaları (Yaklaşık İş Günü: 30 gün ~ 22 bar, 60 gün ~ 44 bar, 90 gün ~ 66 bar)
            # Tam takvim günü hesabı yerine işlem günü (bar) üzerinden gidiyoruz.
            # 1 Ay ~ 21 İş Günü
            # 2 Ay ~ 42 İş Günü
            # 3 Ay ~ 63 İş Günü
            
            fiyat_30g = df['Close'].iloc[-21]
            fiyat_60g = df['Close'].iloc[-42]
            fiyat_90g = df['Close'].iloc[-63]
            
            getiri_30 = ((son_fiyat - fiyat_30g) / fiyat_30g)
            getiri_60 = ((son_fiyat - fiyat_60g) / fiyat_60g)
            getiri_90 = ((son_fiyat - fiyat_90g) / fiyat_90g)
            
            # Teknik
            rsi = ta.rsi(df['Close'], 14).iloc[-1]
            sma50 = ta.sma(df['Close'], 50).iloc[-1]
            
            # Trend Sinyali
            trend = "Yükseliş ↗️" if son_fiyat > sma50 else "Düşüş ↘️"
            
            rapor.append({
                "Hisse": hisse,
                "Fiyat": son_fiyat,
                "30 Günlük": getiri_30,
                "60 Günlük": getiri_60,
                "90 Günlük": getiri_90,
                "RSI": rsi,
                "Trend": trend
            })
        except: continue
        
    return pd.DataFrame(rapor)

# --- DETAYLI ANALİZ MODELLERİ (AI) ---
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
    <div style="background-color:#0E1117; padding:15px; border-radius:10px; border:1px solid #262730; text-align:center;">
        <h3 style="color:#FAFAFA; margin:0;">Mustafa Enes KORKMAZOĞLU</h3>
        <p style="color:#9CA0A6; font-size:12px; margin:5px 0;">NEÜ İktisat 3. Sınıf</p>
        <div style="background-color:#262730; color:#00CC96; padding:5px; border-radius:5px; font-weight:bold; font-size:12px;">
            V9.0 TREASURE
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    col_s1, col_s2 = st.columns(2)
    with col_s1: st.link_button("👔 LinkedIn", "https://www.linkedin.com/in/mustafa-enes-korkmazoglu", use_container_width=True)
    with col_s2: st.link_button("📸 Instagram", "https://www.instagram.com/mustafaenesk_", use_container_width=True)
    st.divider()

    st.header("📲 Menü")
    sayfa = st.radio("Git:", ["💎 Gizli Cevherler (Yan Tahtalar)", "📈 Detaylı Tekil Analiz", "⚖️ Portföy Sihirbazı"])
    
    st.divider()
    
    with st.expander("💰 Hızlı Hesap Makinesi", expanded=False):
        hisse_calc = st.selectbox("Hisse", ["THYAO", "ASELS"] + TUM_HISSELER[:10])
        tutar_calc = st.number_input("Tutar (TL)", 1000, 1000000, 10000, step=1000)
        if st.button("Hesapla"):
            df_c = veri_cek(hisse_calc)
            if not df_c.empty:
                fiyat = df_c['Close'].iloc[-1]
                lot = int(tutar_calc / fiyat)
                st.write(f"📍 Fiyat: {fiyat:.2f} TL")
                st.write(f"📦 Lot: {lot}")

# --- SAYFA 1: GİZLİ CEVHERLER (YENİ ÖZELLİK) ---
if sayfa == "💎 Gizli Cevherler (Yan Tahtalar)":
    st.title("💎 Gözden Kaçan Yıldızlar & Yan Tahtalar")
    st.info("Bu liste BIST 30 harici, büyüme potansiyeli yüksek, enerji ve teknoloji ağırlıklı hisselerden oluşur.")
    
    if st.button("Listeyi Tara ve Analiz Et 🚀", type="primary"):
        with st.spinner("Piyasa taranıyor, geçmiş performanslar hesaplanıyor..."):
            df_tablo = detayli_tarama_yap(GIZLI_CEVHERLER)
            
            if not df_tablo.empty:
                # 30 Günlük Getirisi En Yüksek Olanı En Üste Al
                df_tablo = df_tablo.sort_values(by="30 Günlük", ascending=False)
                
                st.success(f"✅ Tarama Tamamlandı: {len(df_tablo)} Hisse Listelendi")
                
                # --- AKILLI TABLO GÖSTERİMİ ---
                st.dataframe(
                    df_tablo,
                    column_config={
                        "Hisse": st.column_config.TextColumn("Kod"),
                        "Fiyat": st.column_config.NumberColumn("Fiyat (TL)", format="%.2f ₺"),
                        "30 Günlük": st.column_config.ProgressColumn(
                            "1 Aylık Getiri",
                            format="%.2f%%",
                            min_value=-0.5,
                            max_value=0.5,
                        ),
                        "60 Günlük": st.column_config.NumberColumn(
                            "2 Aylık %",
                            format="%.2f%%"
                        ),
                        "90 Günlük": st.column_config.NumberColumn(
                            "3 Aylık %",
                            format="%.2f%%"
                        ),
                        "RSI": st.column_config.NumberColumn(
                            "RSI Gücü",
                            help="30 altı ucuz, 70 üstü pahalı"
                        ),
                        "Trend": st.column_config.TextColumn("Ana Yön")
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=800
                )
            else:
                st.error("Veri çekilemedi.")

# --- SAYFA 2: DETAYLI ANALİZ ---
elif sayfa == "📈 Detaylı Tekil Analiz":
    st.title("📈 Yapay Zeka Destekli Analiz")
    secilen = st.selectbox("Hisse Seçin", TUM_HISSELER)
    
    if st.button("Analizi Başlat"):
        with st.spinner("AI Modelleri Çalışıyor..."):
            df = veri_cek(secilen)
            if not df.empty:
                son = df['Close'].iloc[-1]
                deg = ((son - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100
                
                c1,c2,c3 = st.columns(3)
                c1.metric("Fiyat", f"{son:.2f} ₺", f"%{deg:.2f}")
                
                # Grafik ve AI
                tab1, tab2 = st.tabs(["Grafik", "AI Tahmin"])
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
                        st.write(f"Güven Skoru: %{max(prob)*100:.1f}")
                    
                    # Prophet
                    m = Prophet()
                    m.fit(df.rename(columns={'Date':'ds', 'Close':'y'}))
                    future = m.make_future_dataframe(periods=30)
                    fcast = m.predict(future)
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(x=fcast['ds'], y=fcast['yhat'], line=dict(color='cyan'), name='Tahmin'))
                    fig_p.add_trace(go.Scatter(x=df['Date'], y=df['Close'], line=dict(color='white'), name='Gerçek'))
                    st.plotly_chart(fig_p, use_container_width=True)

# --- SAYFA 3: PORTFÖY ---
elif sayfa == "⚖️ Portföy Sihirbazı":
    st.title("⚖️ Portföy Optimizasyonu (Markowitz)")
    hisseler = st.multiselect("Portföye Eklenecekler", TUM_HISSELER, default=["THYAO", "ASELS", "GARAN"])
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
