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

# --- CANLI VERİ KAZIMA (WEB SCRAPING) ---
@st.cache_data(ttl=43200) # 12 Saatte bir listeyi yeniler
def tum_hisseleri_guncelle():
    """
    Wikipedia'dan BIST'te işlem gören TÜM hisseleri çeker.
    Yeni halka arzlar otomatik olarak buraya düşer.
    """
    yedek_liste = ["ALFAS", "ASTOR", "BIOEN", "BOBET", "BRSAN", "BURCE", "CANTE", "CEMTS", "CVKMD", "CWENE", "DAPGM", "EGEEN", "ENJSA", "EUPWR", "FROTO", "GENIL", "GESAN", "GWIND", "HKTM", "HUNER", "INVEO", "ISMEN", "IZMDC", "JANTS", "KCAER", "KLKIM", "KMPUR", "KNFRT", "KONTR", "MIATK", "MOBTL", "NATEN", "ODAS", "OTKAR", "OYLUM", "OZSUB", "PENTA", "QUAGR", "REEDR", "RUBNS", "SDTTR", "SMRTG", "SNGYO", "SOKM", "SUWEN", "TATGD", "TKFEN", "TTRAK", "VBTYZ", "YEOTK", "YYLGD", "ZOREN"]
    
    try:
        # Wikipedia'daki ana listeyi çek
        url = "https://tr.wikipedia.org/wiki/Borsa_%C4%B0stanbul%27da_i%C5%9Flem_g%C3%B6ren_%C5%9Firketler_listesi"
        tablolar = pd.read_html(url)
        
        tum_kodlar = []
        # Sayfadaki tüm tabloları gez, 'Kod' sütunu olanları al
        for tablo in tablolar:
            if 'Kod' in tablo.columns:
                kodlar = tablo['Kod'].tolist()
                tum_kodlar.extend([str(k).strip().upper() for k in kodlar])
        
        # Tekrarları sil ve sırala
        temiz_liste = sorted(list(set(tum_kodlar)))
        
        if len(temiz_liste) > 50:
            return temiz_liste
        else:
            return yedek_liste
    except:
        return yedek_liste

# Listeleri Oluştur
TUM_HISSELER_CANLI = tum_hisseleri_guncelle()
BIST_30_SABIT = ["AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "DOAS", "EKGYO", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", "KOZAL", "KRDMD", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK"]

# Gizli Cevherler: Tüm hisseler içinden BIST 30'u çıkarıyoruz
GIZLI_CEVHERLER_DINAMIK = [h for h in TUM_HISSELER_CANLI if h not in BIST_30_SABIT]

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
    semboller = [h + ".IS" for h in hisse_listesi]
    
    # Çok fazla hisse varsa (400+), yfinance hata verebilir. İlk 100 tanesini veya rastgele 100 tanesini alalım demo için
    # Veya batch size bölelim. Şimdilik performans için listeden ilk 150 tanesini tarayalım.
    tarama_limiti = hisse_listesi[:150] 
    semboller_limit = [h + ".IS" for h in tarama_limiti]

    try:
        data = yf.download(" ".join(semboller_limit), period="6mo", interval="1d", group_by='ticker', auto_adjust=True, progress=False)
    except:
        return pd.DataFrame()

    rapor = []
    
    for hisse in tarama_limiti:
        try:
            df = data[hisse + ".IS"].copy()
            if df.empty or len(df) < 95: continue 
            
            son_fiyat = df['Close'].iloc[-1]
            
            # Getiriler (İşlem günü bazlı)
            fiyat_30g = df['Close'].iloc[-21]
            fiyat_60g = df['Close'].iloc[-42]
            fiyat_90g = df['Close'].iloc[-63]
            
            getiri_30 = ((son_fiyat - fiyat_30g) / fiyat_30g)
            getiri_60 = ((son_fiyat - fiyat_60g) / fiyat_60g)
            getiri_90 = ((son_fiyat - fiyat_90g) / fiyat_90g)
            
            rsi = ta.rsi(df['Close'], 14).iloc[-1]
            sma50 = ta.sma(df['Close'], 50).iloc[-1]
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

# --- DETAYLI ANALİZ MODELLERİ ---
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
            V10.0 OTO-PİLOT
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    col_s1, col_s2 = st.columns(2)
    with col_s1: st.link_button("👔 LinkedIn", "https://www.linkedin.com/in/mustafa-enes-korkmazoglu", use_container_width=True)
    with col_s2: st.link_button("📸 Instagram", "https://www.instagram.com/mustafaenesk_", use_container_width=True)
    st.divider()

    st.header("📲 Menü")
    # YENİ MENÜ
    sayfa = st.radio("Git:", ["🕵️‍♂️ BIST Keşif Taraması (Canlı)", "📈 Detaylı Tekil Analiz", "⚖️ Portföy Sihirbazı"])
    
    st.divider()
    
    with st.expander("💰 Hızlı Hesap Makinesi", expanded=False):
        hisse_calc = st.selectbox("Hisse", ["THYAO", "ASELS"] + TUM_HISSELER_CANLI[:20])
        tutar_calc = st.number_input("Tutar (TL)", 1000, 1000000, 10000, step=1000)
        if st.button("Hesapla"):
            df_c = veri_cek(hisse_calc)
            if not df_c.empty:
                fiyat = df_c['Close'].iloc[-1]
                lot = int(tutar_calc / fiyat)
                st.write(f"📍 Fiyat: {fiyat:.2f} TL")
                st.write(f"📦 Lot: {lot}")

# --- SAYFA 1: OTOMATİK KEŞİF (YENİ) ---
if sayfa == "🕵️‍♂️ BIST Keşif Taraması (Canlı)":
    st.title("🕵️‍♂️ BIST Canlı Keşif Ekranı")
    st.info(f"Şu an sistemde **{len(TUM_HISSELER_CANLI)} adet** hisse kayıtlı. BIST 30 harici **{len(GIZLI_CEVHERLER_DINAMIK)} adet** potansiyel hisse taranmayı bekliyor.")
    
    st.warning("⚠️ Not: Tam liste taraması (400+ hisse) uzun sürebileceği için, performans açısından ilk 150 hisse taranacaktır.")

    if st.button("Taramayı Başlat ve Yenileri Bul 🚀", type="primary"):
        with st.spinner("Wikipedia listesi güncelleniyor ve piyasa taranıyor..."):
            
            # Dinamik listeyi tarıyoruz
            df_tablo = detayli_tarama_yap(GIZLI_CEVHERLER_DINAMIK)
            
            if not df_tablo.empty:
                # 30 Günlük Getirisi En Yüksek Olanı En Üste Al
                df_tablo = df_tablo.sort_values(by="30 Günlük", ascending=False)
                
                st.success(f"✅ Tarama Tamamlandı! İşte Sonuçlar:")
                
                # --- AKILLI TABLO ---
                st.dataframe(
                    df_tablo,
                    column_config={
                        "Hisse": st.column_config.TextColumn("Kod"),
                        "Fiyat": st.column_config.NumberColumn("Fiyat (TL)", format="%.2f ₺"),
                        "30 Günlük": st.column_config.ProgressColumn(
                            "1 Ay Getiri",
                            format="%.2f%%",
                            min_value=-0.5,
                            max_value=0.5,
                        ),
                        "60 Günlük": st.column_config.NumberColumn(
                            "2 Ay %",
                            format="%.2f%%"
                        ),
                        "90 Günlük": st.column_config.NumberColumn(
                            "3 Ay %",
                            format="%.2f%%"
                        ),
                        "RSI": st.column_config.NumberColumn(
                            "RSI",
                            help="30 altı fırsat olabilir"
                        ),
                        "Trend": st.column_config.TextColumn("Yön")
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=800
                )
            else:
                st.error("Veri çekilemedi veya piyasa kapalı.")

# --- SAYFA 2: DETAYLI ANALİZ ---
elif sayfa == "📈 Detaylı Tekil Analiz":
    st.title("📈 Yapay Zeka Destekli Analiz")
    # Listeyi Canlı Listeden Çekiyoruz
    secilen = st.selectbox("Hisse Seçin", TUM_HISSELER_CANLI)
    
    if st.button("Analizi Başlat"):
        with st.spinner("AI Modelleri Çalışıyor..."):
            df = veri_cek(secilen)
            if not df.empty:
                son = df['Close'].iloc[-1]
                deg = ((son - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100
                
                c1,c2,c3 = st.columns(3)
                c1.metric("Fiyat", f"{son:.2f} ₺", f"%{deg:.2f}")
                
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
    hisseler = st.multiselect("Portföye Eklenecekler", TUM_HISSELER_CANLI, default=["THYAO", "ASELS", "GARAN"])
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
