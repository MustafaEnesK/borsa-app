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

# --- SESSION STATE (HAFIZA) AYARLARI ---
# Sayfa yenilense bile analizin kaybolmaması için hafıza oluşturuyoruz
if 'analiz_aktif' not in st.session_state:
    st.session_state.analiz_aktif = False
if 'secilen_hisse' not in st.session_state:
    st.session_state.secilen_hisse = ""

# --- LİSTELER ---
BIST_30 = [
    "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "DOAS",
    "EKGYO", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR",
    "KCHOL", "KONTR", "KOZAL", "KRDMD", "OYAKC", "PETKM", "PGSUS", "SAHOL",
    "SASA", "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK"
]
BIST_100_EK = ["AEFES", "AGHOL", "AHGAZ", "AKFGY", "AKSA", "ALGYO", "BERA", "CANTE", "CIMSA", "CWENE", "EGEEN", "ENJSA", "EUPWR", "GESAN", "GWIND", "HALKB", "ISGYO", "IZMDC", "KLSER", "MAVI", "MGROS", "MIATK", "ODAS", "OTKAR", "QUAGR", "REEDR", "SKBNK", "SMRTG", "SOKM", "TAVHL", "TKFEN", "TTKOM", "ULKER", "VAKBN", "VESBE", "YEOTK", "YYLGD", "ZOREN"]
BIST_100 = sorted(list(set(BIST_30 + BIST_100_EK)))

# --- FONKSİYONLAR ---
@st.cache_data(ttl=600) 
def verileri_getir(sembol):
    if not sembol.endswith(".IS"):
        sembol += ".IS"
    df = yf.download(sembol, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df.empty: return pd.DataFrame()
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
            "Temettü": info.get("dividendYield", 0)
        }
    except:
        return None

def prophet_tahmin(df, gun_sayisi=65):
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
        return feed.entries[:4]
    except:
        return []

# --- YAN MENÜ ---
with st.sidebar:
    st.markdown("""
    <div style="background-color:#1E1E1E; padding:15px; border-radius:10px; border:1px solid #333;">
        <h3 style="color:white; margin:0;">Mustafa Enes KORKMAZOĞLU</h3>
        <p style="color:#888; margin:5px 0;">🎓 NEÜ İktisat 3. Sınıf</p>
        <p style="color:#00CC96; font-weight:bold;">🚀 Borsa Simülasyonu PRO</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") 
    st.link_button("👔 LinkedIn Profilim", "https://www.linkedin.com/in/mustafa-enes-korkmazoglu", use_container_width=True)
    st.link_button("📸 Instagram: @mustafaenesk_", "https://www.instagram.com/mustafaenesk_", use_container_width=True)

    st.divider()

    st.header("🔍 Hisse Seçimi")
    secim_modu = st.radio("Liste Seçiniz:", ["Manuel Arama", "BIST 30", "BIST 100"])
    
    if secim_modu == "Manuel Arama":
        hisse_input = st.text_input("Hisse Kodu Girin", "THYAO").upper()
    elif secim_modu == "BIST 30":
        hisse_input = st.selectbox("BIST 30 Hissesi Seç", BIST_30)
    else:
        hisse_input = st.selectbox("BIST 100 Hissesi Seç", BIST_100)
    
    st.subheader("🎨 Grafik Seçenekleri")
    goster_sma50 = st.checkbox("SMA 50 (Turuncu)", value=True)
    goster_sma200 = st.checkbox("SMA 200 (Mavi)", value=True)
    goster_bollinger = st.checkbox("Bollinger Bantları", value=False)
    
    st.divider()
    # Butona basınca Session State güncellenir
    if st.button("Analizi Başlat 🚀", type="primary", use_container_width=True):
        st.session_state.analiz_aktif = True
        st.session_state.secilen_hisse = hisse_input
        st.cache_data.clear()

# --- ANA SAYFA ---
st.title("📈 Borsa İstanbul Yapay Zeka Analisti")

# Eğer butona basıldıysa VE hisse seçildiyse ekranı göster
if st.session_state.analiz_aktif:
    hisse_kodu = st.session_state.secilen_hisse
    saf_kod = hisse_kodu.replace(".IS", "")
    
    st.caption(f"Analiz Edilen Hisse: **{hisse_kodu}**")
    
    with st.spinner('Veriler ve Yapay Zeka Tahminleri Hazırlanıyor...'):
        df = verileri_getir(hisse_kodu)
        info = temel_bilgileri_getir(hisse_kodu)
        
        if df.empty:
            st.error("Veri bulunamadı! Lütfen kodu kontrol edin.")
        else:
            # ÜST KARTLAR
            son_fiyat = df['Close'].iloc[-1]
            degisim = ((son_fiyat - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Son Fiyat", f"{son_fiyat:.2f} ₺", f"%{degisim:.2f}")
            if info:
                temettu = info['Temettü']
                fmt_temettu = f"%{temettu*100:.2f}" if isinstance(temettu, float) else "-"
                c2.metric("F/K", f"{info['F/K']}")
                c3.metric("PD/DD", f"{info['PD/DD']}")
                c4.metric("Temettü", fmt_temettu)

            # GRAFİK
            st.subheader(f"📊 {saf_kod} Teknik Analiz")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'))
            if goster_sma50:
                df['SMA50'] = ta.sma(df['Close'], length=50)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], line=dict(color='orange', width=1), name='SMA 50'))
            if goster_sma200:
                df['SMA200'] = ta.sma(df['Close'], length=200)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA200'], line=dict(color='blue', width=1), name='SMA 200'))
            if goster_bollinger:
                bb = ta.bbands(df['Close'], length=20, std=2)
                if bb is not None:
                    fig.add_trace(go.Scatter(x=df['Date'], y=bb[bb.columns[0]], line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dot'), name='Alt Bant'))
                    fig.add_trace(go.Scatter(x=df['Date'], y=bb[bb.columns[2]], line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', name='Üst Bant'))
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # SİMÜLATÖR VE TAHMİN
            st.divider()
            col_tahmin, col_sim = st.columns([1, 1])

            try:
                fcast = prophet_tahmin(df, 65)
                future_data = fcast[fcast['ds'] > df['Date'].iloc[-1]].copy()
                future_data.reset_index(drop=True, inplace=True)
                
                t15 = future_data.iloc[14]['yhat'] if len(future_data) > 14 else 0
                t30 = future_data.iloc[29]['yhat'] if len(future_data) > 29 else 0
                t60 = future_data.iloc[59]['yhat'] if len(future_data) > 59 else 0

                with col_tahmin:
                    st.subheader("🔮 Yapay Zeka Beklentisi")
                    st.info("AI modelinin geçmiş verilere dayanarak yaptığı fiyat öngörüsüdür.")
                    t_data = {
                        "Vade": ["15 Gün Sonra", "30 Gün Sonra", "60 Gün Sonra"],
                        "Tahmini Fiyat": [f"{t15:.2f} ₺", f"{t30:.2f} ₺", f"{t60:.2f} ₺"],
                        "Beklenen Değişim": [
                            f"%{((t15-son_fiyat)/son_fiyat)*100:.2f}",
                            f"%{((t30-son_fiyat)/son_fiyat)*100:.2f}",
                            f"%{((t60-son_fiyat)/son_fiyat)*100:.2f}"
                        ]
                    }
                    st.table(pd.DataFrame(t_data))
                    
                    fig_mini = go.Figure()
                    fig_mini.add_trace(go.Scatter(x=future_data['ds'], y=future_data['yhat'], mode='lines', line=dict(color='#00CC96', width=2), name='Tahmin'))
                    fig_mini.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig_mini, use_container_width=True)

                with col_sim:
                    st.subheader("💰 Yatırım Simülatörü")
                    st.success("Cebindeki parayı gir, Enter'a bas.")
                    
                    # BURASI DÜZELTİLDİ: Session State sayesinde burası çalışınca sayfa kapanmayacak
                    ana_para = st.number_input("Yatırılacak Tutar (TL)", min_value=1000, value=10000, step=1000)
                    
                    st.write(f"**{ana_para:,.0f} TL** ile bugün {saf_kod} alınırsa:")
                    
                    kar15 = ana_para * (t15 / son_fiyat)
                    kar30 = ana_para * (t30 / son_fiyat)
                    kar60 = ana_para * (t60 / son_fiyat)
                    
                    c_sim1, c_sim2, c_sim3 = st.columns(3)
                    
                    renk15 = "off" if kar15 > ana_para else "inverse"
                    c_sim1.metric("15 Gün", f"{kar15:,.0f} ₺", f"{kar15-ana_para:,.0f} ₺", delta_color=renk15)
                    
                    renk30 = "off" if kar30 > ana_para else "inverse"
                    c_sim2.metric("30 Gün", f"{kar30:,.0f} ₺", f"{kar30-ana_para:,.0f} ₺", delta_color=renk30)
                    
                    renk60 = "off" if kar60 > ana_para else "inverse"
                    c_sim3.metric("60 Gün", f"{kar60:,.0f} ₺", f"{kar60-ana_para:,.0f} ₺", delta_color=renk60)

            except Exception as e:
                st.warning(f"Simülasyon hesaplanamadı: {e}")

            # HABERLER
            st.divider()
            st.subheader("📰 Güncel Haberler")
            cols = st.columns(2)
            news = haberleri_getir(saf_kod)
            if news:
                for i, h in enumerate(news):
                    cols[i % 2].info(f"[{h.title}]({h.link})")
else:
    st.info("👈 Analize başlamak için sol menüden bir hisse seçin ve 'Analizi Başlat' butonuna basın.")
