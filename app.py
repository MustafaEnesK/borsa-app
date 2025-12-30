import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import feedparser
import matplotlib.pyplot as plt

# --- SAYFA AYARLARI (MOBİL İÇİN) ---
st.set_page_config(page_title="Cep Borsa", page_icon="📈", layout="centered")

# --- BAŞLIK ---
st.title("📈 Cep Borsa Analisti")
st.caption("Yapay Zeka Destekli BIST Sinyalleri")

# --- KULLANICI GİRİŞİ ---
hisse_kodu = st.text_input("Hisse Kodu Girin (Örn: ASELS)", "THYAO")
analiz_butonu = st.button("Analiz Et 🚀", type="primary")

# --- FONKSİYONLAR ---
def verileri_getir(sembol):
    if not sembol.endswith(".IS"):
        sembol += ".IS"
    try:
        # Son 6 ayın verisi yeterli
        df = yf.download(sembol, period="6mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

def haberleri_getir(saf_kod):
    try:
        rss_url = f"https://news.google.com/rss/search?q={saf_kod}+hisse&hl=tr&gl=TR&ceid=TR:tr"
        feed = feedparser.parse(rss_url)
        return feed.entries[:3]
    except:
        return []

# --- ANALİZ MOTORU ---
if analiz_butonu:
    saf_kod = hisse_kodu.replace(".IS", "").upper()
    
    with st.spinner(f'{saf_kod} verileri çekiliyor...'):
        df = verileri_getir(hisse_kodu)
        
        if df.empty or len(df) < 50:
            st.error("Veri bulunamadı! Kodu doğru yazdığınızdan emin olun.")
        else:
            # İndikatörler
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['SMA50'] = ta.sma(df['Close'], length=50)
            
            son_fiyat = df['Close'].iloc[-1]
            son_rsi = df['RSI'].iloc[-1]
            son_sma = df['SMA50'].iloc[-1]
            
            # --- SKORLAMA ---
            skor = 50
            nedenler = []
            
            # RSI Yorumu
            if son_rsi < 35: 
                skor += 20
                nedenler.append("RSI Dipte (Alım Fırsatı Olabilir)")
            elif son_rsi > 70: 
                skor -= 20
                nedenler.append("RSI Zirvede (Satış Gelebilir)")
            
            # Trend Yorumu
            if son_fiyat > son_sma: 
                skor += 15
                nedenler.append("Trend: YÜKSELİŞ (Fiyat > SMA50)")
            else: 
                skor -= 15
                nedenler.append("Trend: DÜŞÜŞ (Fiyat < SMA50)")

            # --- EKRANA BASMA (MOBİL GÖRÜNÜM) ---
            
            # 1. KUTULAR
            col1, col2 = st.columns(2)
            col1.metric("Fiyat", f"{son_fiyat:.2f} ₺")
            col2.metric("RSI", f"{son_rsi:.1f}", delta_color="off")
            
            # 2. KARAR
            st.divider()
            st.subheader("🤖 Simülasyon Kararı")
            
            if skor >= 70:
                st.success(f"GÜÇLÜ AL (Skor: {skor})")
            elif skor >= 55:
                st.info(f"AL / TUT (Skor: {skor})")
            elif skor <= 30:
                st.error(f"SAT (Skor: {skor})")
            else:
                st.warning(f"NÖTR / İZLE (Skor: {skor})")
                
            for neden in nedenler:
                st.write(f"- {neden}")

            # 3. GRAFİK
            st.divider()
            st.write("📊 **Teknik Görünüm**")
            fig, ax = plt.subplots(figsize=(8, 4)) # Mobilde küçük grafik daha iyi
            ax.plot(df.index, df['Close'], label='Fiyat', color='black')
            ax.plot(df.index, df['SMA50'], label='Ortalama (50)', color='orange', linestyle='--')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # 4. HABERLER
            st.divider()
            st.write(f"🌍 **{saf_kod} Haberleri**")
            haberler = haberleri_getir(saf_kod)
            if haberler:
                for h in haberler:
                    st.markdown(f"• [{h.title}]({h.link})")
            else:
                st.write("Haber akışı yok.")
                