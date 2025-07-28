import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from streamlit_autorefresh import st_autorefresh

# --- Asetukset ---
API_KEY = "d224d79r01qt8676madgd224d79r01qt8676mae0"
st.set_page_config(layout="centered")
st_autorefresh(interval=5 * 60 * 1000, key="auto_refresh")
st.title("📈 Osakeagentti – Usean osakkeen vertailu + RSI/MACD")

# --- Syötteet ---
symbols_input = st.text_input("Syötä osaketunnukset pilkulla erotettuna (esim. TSLA,AAPL,MSFT):", "TSLA")
risk = st.selectbox("Valitse riskitaso:", ["matala", "keskitaso", "korkea"])
days_ahead = st.slider("Valitse ennustepäivien määrä:", 3, 10, 5)

symbols = [sym.strip().upper() for sym in symbols_input.split(",") if sym.strip()]

@st.cache_data
def fetch_news(symbol, api_key):
    url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-07-01&to=2024-07-28&token={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()[:5]
    return []

@st.cache_resource
def load_finbert_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

def analyze_sentiment_finbert(news_list):
    tokenizer, model = load_finbert_model()
    sentiments = []
    for item in news_list:
        text = item.get('headline', '')
        if not text:
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs).item()
        score = probs[0][label].item()
        sentiment_value = {-1: -score, 0: 0, 1: score}[label - 1]
        sentiments.append(sentiment_value)
    return sum(sentiments) / len(sentiments) if sentiments else 0

def compute_indicators(df):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

@st.cache_data
def load_data(symbol, days_ahead, sentiment_score):
    data = yf.download(symbol, period="2y", interval="1d")
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['Return'] = data['Close'].pct_change()
    data['Sentiment'] = sentiment_score
    data = compute_indicators(data)
    data = data.dropna()
    data['Target'] = np.where(data['Close'].shift(-days_ahead) > data['Close'], 1, 0)
    return data

st.write("---")
st.info(
    """
    **Tekoälyn päätöksen perusteet:**

    - Käytämme historiallisia osakehinnan piirteitä: SMA_10, SMA_50, tuotto (%).
    - Uutisten sentimentti arvioidaan FinBERT-mallilla (positiivinen / negatiivinen vaikutus).
    - RSI ja MACD indikaattorit auttavat tunnistamaan yliostettuja tai ylimyytyjä tilanteita.
    - Malli antaa luottamusarvon, jota verrataan riskitasoon.
    - Suositukset ovat: 📈 OSTA, 🤔 PIDÄ (epävarma), 📉 MYY.
    """
)

results = []
for symbol in symbols:
    st.subheader(f"📊 Analyysi: {symbol}")

    news = fetch_news(symbol, API_KEY)
    if news:
        st.write("🗞️ Viimeisimmät uutiset:")
        for n in news:
            st.write(f"- {n['datetime'][:10]}: {n['headline']}")
    else:
        st.write("Ei uutisia saatavilla.")

    sentiment_score = analyze_sentiment_finbert(news)
    st.write(f"Sentimenttipiste (FinBERT): {sentiment_score:.3f}")

    data = load_data(symbol, days_ahead, sentiment_score)
    features = ['SMA_10', 'SMA_50', 'Return', 'Sentiment', 'RSI', 'MACD', 'Signal_Line']
    X = data[features]
    y = data['Target']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[:-10], y[:-10])

    latest = X.iloc[[-1]]
    proba = model.predict_proba(latest)[0]
    confidence = abs(proba[1] - proba[0])
    prediction = model.predict(latest)[0]

    thresholds = {"matala": 0.05, "keskitaso": 0.1, "korkea": 0.2}
    threshold = thresholds[risk]

    if confidence < threshold:
        suggestion = "🤔 PIDÄ (epävarma signaali)"
    elif prediction == 1:
        suggestion = "📈 OSTA"
    else:
        suggestion = "📉 MYY"

    st.write(f"**Agentin suositus:** {suggestion}")
    st.write(f"Luottamus: `{confidence:.2f}`")
    st.write(f"**RSI:** {data['RSI'].iloc[-1]:.2f}")
    st.write(f"**MACD:** {data['MACD'].iloc[-1]:.2f} | Signaali: {data['Signal_Line'].iloc[-1]:.2f}")

    results.append((symbol, suggestion, confidence))

# Yhteenveto
st.write("---")
st.subheader("📋 Yhteenveto suosituksista")
for symbol, suggestion, confidence in results:
    st.write(f"{symbol}: {suggestion} (luottamus {confidence:.2f})")
