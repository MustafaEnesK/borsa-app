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

# --- 1. GÜVENLİK KAPISI (YASAL UYARI) ---
if 'yasal_kabul' not in st.session_state: st.session_state.yasal_kabul = False

if not st.session_state.yasal_kabul:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="background-color:#262730; padding:30px; border-radius:15px; border:2px solid #FF4B4B; text-align:center;">
            <h1 style="color:#FF4B4B;">⚠️ YASAL UYARI</h1>
            <hr>
            <p><strong>BorsApp</strong> eğitim amaçlı bir simülasyondur.</p>
            <p>Buradaki veriler, AI tahminleri ve sinyaller <strong>YATIRIM TAVSİYESİ DEĞİLDİR.</strong></p>
            <p>Borsa risk içerir. Tüm sorumluluk kullanıcıya aittir.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("✅ Okudum, Riskleri Kabul Ediyorum", type="primary", use_container_width=True):
            st.session_state.yasal_kabul = True
            st.rerun()
    st.stop()

# --- 2. AYARLAR VE LİSTELER ---
if 'analiz_aktif' not in st.session_state: st.session_state.analiz_aktif = False
if 'secilen_hisse' not in st.session_state: st.session_state.secilen_hisse = ""

@st.cache_data(ttl=43200)
def listeleri_hazirla():
    bist30 = sorted(["AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "DOAS", "EKGYO", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", "KOZAL", "KRDMD", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK"])
    yedek_bist100 = bist30 + ["AEFES", "AGHOL", "AHGAZ", "AKFGY", "AKSA", "ALGYO", "BERA", "CANTE", "CIMSA", "EGEEN", "ENJSA", "EUPWR", "GESAN", "GWIND", "HALKB", "ISGYO", "IZMDC", "KCAER", "MAVI", "MGROS", "MIATK", "ODAS", "OTKAR", "QUAGR", "REEDR", "SKBNK", "SMRTG", "SOKM", "TAVHL", "TKFEN", "TTKOM", "ULKER", "VAKBN", "VESBE", "YEOTK", "YYLGD", "ZOREN"]
    try:
        url = "https://tr.wikipedia.org/wiki/Borsa_%C4%B0stanbul%27da_i%C5%9Flem_g%C3%B6ren_%C5%9Firketler_listesi"
        tablolar = pd.read_html(url)
        tum = []
        for t in tablolar:
            if 'Kod' in t.columns: tum.extend([str(k).strip().upper() for k in t['Kod'].tolist()])
        final = sorted(list(set(tum)))
        return bist30, (final if len(final)>50 else yedek_bist100)
    except: return bist30, yedek_bist100

BIST_30_LISTESI, TUM_HISSELER = listeleri_hazirla()
GIZLI_CEVHERLER = [h for h in TUM_HISSELER if h not in BIST_30_LISTESI]

# --- 3. FONKSİYONLAR ---
@st.cache_data(ttl=600) 
def veri_cek(kod):
    if not kod.endswith(".IS"): kod += ".IS"
    df = yf.download(kod, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

def sirket_ismini_bul(kod):
    try:
        if not kod.endswith(".IS"): kod += ".IS"
        return yf.Ticker(kod).info.get('longName', kod)
    except: return kod

def haberleri_getir(kod):
    try:
        rss = f"https://news.google.com/rss/search?q={kod}+hisse&hl=tr&gl=TR&ceid=TR:tr"
        return feedparser.parse(rss).entries[:5]
    except: return []

@st.cache_data(ttl=120)
def canli_piyasa_tablosu(hisse_listesi):
    liste = hisse_listesi[:100]
    semboller = " ".join([h + ".IS" for h in liste])
    try:
        data = yf.download(semboller, period="5d", interval="1d", group_by='ticker', auto_adjust=True, progress=False)
    except: return pd.DataFrame()
    rapor = []
    for h in liste:
        try:
            df = data[h + ".IS"]
            if df.empty: continue
            son = df['Close'].iloc[-1]
            deg = ((son - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100
            hacim = df['Volume'].iloc[-1]
            durum = "NÖTR ⚪"
            if deg > 3: durum = "GÜÇLÜ ALICI 🟢🟢"
            elif deg > 0: durum = "POZİTİF 🟢"
            elif deg < -3: durum = "GÜÇLÜ SATICI 🔴🔴"
            elif deg < 0: durum = "NEGATİF 🔴"
            rapor.append({"Kod": h, "Fiyat": son, "Değişim %": deg, "Hacim": hacim, "Durum": durum})
        except: continue
    return pd.DataFrame(rapor)

@st.cache_data(ttl=3600)
def detayli_kesif_taramasi(hisse_listesi):
    liste = hisse_listesi[:150]
    semboller = " ".join([h + ".IS" for h in liste])
    try:
        data = yf.download(semboller, period="6mo", interval="1d", group_by='ticker', auto_adjust=True, progress=False)
    except: return pd.DataFrame()
    rapor = []
    for h in liste:
        try:
            df = data[h + ".IS"]
            if df.empty or len(df)<70: continue
            son = df['Close'].iloc[-1]
            g30 = ((son - df['Close'].iloc[-22])/df['Close'].iloc[-22])
            g60 = ((son - df['Close'].iloc[-44])/df['Close'].iloc[-44])
            rsi = ta.rsi(df['Close'], 14).iloc[-1]
            rapor.append({
                "Hisse": h, "Fiyat": son, "30G Getiri": g30, "60G Getiri": g60, "RSI": rsi,
                "Sinyal": "AL 🟢" if rsi < 30 else ("SAT 🔴" if rsi > 70 else "NÖTR ⚪")
            })
        except: continue
    return pd.DataFrame(rapor)

# --- AI MODELLERİ ---
def xgboost_analiz(df):
    try:
        data = df.copy()
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
        return model.predict(last)[0], model.predict_proba(last)[0]
    except: return None, None

def prophet_tahmin(df, gun=30):
    try:
        df_p = df[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
        m = Prophet()
        m.fit(df_p)
        fut = m.make_future_dataframe(periods=gun)
        return m.predict(fut)
    except: return pd.DataFrame()

def markowitz(hisseler, butce):
    if len(hisseler)<2: return None, "Min 2 hisse."
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

# --- ARAYÜZ (SIDEBAR) ---
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
    sayfa = st.radio("Modül Seçiniz:", 
        ["🔎 Detaylı Hisse Analizi", 
         "🕵️‍♂️ Keşif Taraması (Gizli Fırsatlar)", 
         "🏆 BIST 30 Canlı", 
         "💯 BIST 100 Canlı", 
         "⚖️ Akıllı Portföy"])
    
    st.divider()
    
    # GLOBAL HESAP MAKİNESİ
    with st.expander("💰 Hızlı Hesap Makinesi", expanded=False):
        h_calc = st.selectbox("Hisse", ["THYAO"] + BIST_30_LISTESI)
        t_calc = st.number_input("Tutar", 1000, 1000000, 10000)
        if st.button("Lot Hesapla"):
            d = veri_cek(h_calc)
            if not d.empty:
                p = d['Close'].iloc[-1]
                st.write(f"Fiyat: {p:.2f} TL")
                st.write(f"Lot: {int(t_calc/p)}")

# ==============================================================================
# SAYFA 1: DETAYLI HİSSE ANALİZİ (GÜNCELLENMİŞ SİMÜLATÖR İLE)
# ==============================================================================
if sayfa == "🔎 Detaylı Hisse Analizi":
    st.title("🔎 Profesyonel Hisse Analizi")
    
    c1, c2 = st.columns([3,1])
    with c1: kod_giris = st.text_input("Hisse Kodu (Örn: EBEBK, KONTR)", "THYAO").upper()
    with c2: 
        st.write("")
        st.write("")
        btn = st.button("ANALİZ ET 🚀", type="primary", use_container_width=True)

    if btn or kod_giris:
        with st.spinner("Yapay Zeka verileri işliyor..."):
            df = veri_cek(kod_giris)
            if not df.empty:
                isim = sirket_ismini_bul(kod_giris)
                son = df['Close'].iloc[-1]
                deg = ((son - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100
                renk = "green" if deg > 0 else "red"
                
                st.markdown(f"### {isim}")
                st.markdown(f"# {son:.2f} ₺ <span style='color:{renk}; font-size:24px'>%{deg:.2f}</span>", unsafe_allow_html=True)
                
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Genel Bakış", "🧠 Yapay Zeka", "📰 Haberler", "📅 Veri Tablosu"])
                
                with tab1: # Grafik & Temel
                    st.plotly_chart(go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])]).update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)
                    try:
                        info = yf.Ticker(kod_giris+".IS").info
                        c_1, c_2, c_3, c_4 = st.columns(4)
                        c_1.metric("F/K", f"{info.get('trailingPE','-'):.2f}" if info.get('trailingPE') else '-')
                        c_2.metric("PD/DD", f"{info.get('priceToBook','-'):.2f}" if info.get('priceToBook') else '-')
                        c_3.metric("Hacim", f"{df['Volume'].iloc[-1]:,}")
                        c_4.metric("Zirve", f"{info.get('fiftyTwoWeekHigh','-')} ₺")
                    except: pass

                with tab2: # AI
                    c_ai1, c_ai2 = st.columns([1,2])
                    pred, prob = xgboost_analiz(df)
                    with c_ai1:
                        st.subheader("AI Sinyali")
                        if pred is not None:
                            guven = max(prob)*100
                            if pred==1: st.success(f"YÜKSELİŞ (Güven: %{guven:.1f})")
                            else: st.error(f"DÜŞÜŞ (Güven: %{guven:.1f})")
                        else: st.warning("Veri yetersiz.")
                    with c_ai2:
                        st.subheader("Gelecek Tahmini")
                        fc = prophet_tahmin(df)
                        if not fc.empty:
                            fig_p = go.Figure()
                            fig_p.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Gerçek', line=dict(color='white')))
                            fut = fc[fc['ds']>df['Date'].iloc[-1]]
                            fig_p.add_trace(go.Scatter(x=fut['ds'], y=fut['yhat'], name='Tahmin', line=dict(color='cyan', dash='dash')))
                            st.plotly_chart(fig_p.update_layout(height=350, template="plotly_dark"), use_container_width=True)

                with tab3: # Haberler
                    news = haberleri_getir(kod_giris)
                    if news:
                        for n in news: st.info(f"[{n.title}]({n.link})")
                    else: st.write("Haber yok.")

                with tab4: # Tablo
                    st.dataframe(df.tail(30).sort_values(by="Date", ascending=False), use_container_width=True)

                # --- YENİLENEN SİTE İÇİ HESAP MAKİNESİ ---
                st.divider()
                st.subheader(f"🧮 {kod_giris} İçin AI Getiri Simülatörü")
                
                # Kullanıcı Girişleri
                c_sim1, c_sim2, c_sim3 = st.columns(3)
                with c_sim1:
                    yatirim_tutar = st.number_input("Yatırılacak Tutar (TL)", min_value=1000, value=10000, step=1000)
                with c_sim2:
                    vade_secimi = st.selectbox("Yatırım Vadesi Seç", ["1 Ay", "3 Ay", "6 Ay", "1 Yıl"])
                    gun_cevir = {"1 Ay": 30, "3 Ay": 90, "6 Ay": 180, "1 Yıl": 365}
                
                # Hesaplama Butonu
                with c_sim3:
                    st.write("") # Hizalama boşluğu
                    st.write("")
                    hesapla_btn = st.button("Geleceği Hesapla 🔮", type="primary", use_container_width=True)

                if hesapla_btn:
                    with st.spinner("Yapay Zeka Geleceği Hesaplıyor..."):
                        # Prophet ile vade sonu tahmini
                        tahmin_df = prophet_tahmin(df, gun_cevir[vade_secimi])
                        
                        if not tahmin_df.empty:
                            # Hesaplamalar
                            gelecek_fiyat = tahmin_df['yhat'].iloc[-1]
                            lot_sayisi = int(yatirim_tutar / son)
                            gelecek_tutar = lot_sayisi * gelecek_fiyat
                            kar_zarar = gelecek_tutar - yatirim_tutar
                            yuzde_kar = (kar_zarar / yatirim_tutar) * 100
                            
                            # Sonuç Kartları
                            st.write("---")
                            st.info(f"📆 **Vade:** {vade_secimi} Sonra")
                            
                            res1, res2, res3 = st.columns(3)
                            res1.metric("Tahmini Fiyat", f"{gelecek_fiyat:.2f} TL", f"%{yuzde_kar:.2f}")
                            res2.metric("Gelecek Bakiye", f"{gelecek_tutar:,.0f} TL")
                            
                            renk_sonuc = "normal" if kar_zarar > 0 else "inverse"
                            res3.metric("Net Kar/Zarar", f"{kar_zarar:,.0f} TL", delta_color=renk_sonuc)
                        else:
                            st.error("Tahmin verisi oluşturulamadı.")

            else: st.error("Hisse bulunamadı.")

# ==============================================================================
# SAYFA 2: KEŞİF TARAMASI
# ==============================================================================
elif sayfa == "🕵️‍♂️ Keşif Taraması (Gizli Fırsatlar)":
    st.title("🕵️‍♂️ BIST 100 Dışı Fırsatlar")
    st.info("BIST 30 harici, büyüme potansiyeli olan hisseler taranıyor.")
    if st.button("Taramayı Başlat 🚀", type="primary"):
        with st.spinner("Piyasa taranıyor..."):
            df_kesif = detayli_kesif_taramasi(GIZLI_CEVHERLER)
            if not df_kesif.empty:
                df_kesif = df_kesif.sort_values(by="30G Getiri", ascending=False)
                st.dataframe(
                    df_kesif,
                    column_config={
                        "Hisse": st.column_config.TextColumn("Kod"),
                        "Fiyat": st.column_config.NumberColumn("Fiyat", format="%.2f ₺"),
                        "30G Getiri": st.column_config.ProgressColumn("1 Ay %", format="%.2f%%", min_value=-0.5, max_value=0.5),
                        "60G Getiri": st.column_config.NumberColumn("2 Ay %", format="%.2f%%"),
                        "RSI": st.column_config.NumberColumn("RSI"),
                    }, hide_index=True, use_container_width=True, height=800
                )
            else: st.error("Veri alınamadı.")

# ==============================================================================
# SAYFA 3 & 4: CANLI PİYASA
# ==============================================================================
elif sayfa == "🏆 BIST 30 Canlı":
    st.title("🏆 BIST 30 Canlı Takip")
    if st.button("Yenile"): st.cache_data.clear()
    with st.spinner("Yükleniyor..."):
        df_30 = canli_piyasa_tablosu(BIST_30_LISTESI)
        if not df_30.empty:
            st.dataframe(df_30.sort_values(by="Değişim %", ascending=False), 
                         column_config={"Fiyat": st.column_config.NumberColumn(format="%.2f ₺"), "Değişim %": st.column_config.NumberColumn(format="%.2f%%")},
                         hide_index=True, use_container_width=True, height=800)

elif sayfa == "💯 BIST 100 Canlı":
    st.title("💯 BIST 100 Canlı Takip")
    if st.button("Yenile"): st.cache_data.clear()
    with st.spinner("Yükleniyor..."):
        df_100 = canli_piyasa_tablosu(TUM_HISSELER)
        if not df_100.empty:
            st.dataframe(df_100.sort_values(by="Değişim %", ascending=False), 
                         column_config={"Fiyat": st.column_config.NumberColumn(format="%.2f ₺"), "Değişim %": st.column_config.NumberColumn(format="%.2f%%")},
                         hide_index=True, use_container_width=True, height=800)

# ==============================================================================
# SAYFA 5: PORTFÖY
# ==============================================================================
elif sayfa == "⚖️ Akıllı Portföy":
    st.title("⚖️ Markowitz Portföy Optimizasyonu")
    hisseler = st.multiselect("Portföy Oluştur", TUM_HISSELER, default=["THYAO", "ASELS", "GARAN"])
    butce = st.number_input("Bütçe (TL)", 1000, 1000000, 50000)
    if st.button("Optimize Et"):
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
