import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import torch
from transformers import pipeline

# --- Ladataan sentimenttipipeline kerran ---
sentiment_pipeline = pipeline("sentiment-analysis")

# --- Asetukset ---
API_KEY = "d224d79r01qt8676madgd224d79r01qt8676mae0"  # Finnhub API-avain

st.set_page_config(layout="centered")
st.title("📈 Osakeagentti – Uutisdata + Riskiprofiili")

# --- Käyttäjän syötteet ---
symbol = st.text_input("Syötä osaketunnus (esim. TSLA, AAPL):", "TSLA")
risk = st.selectbox("Valitse riskitaso:", ["matala", "keskitaso", "korkea"])
days_ahead = st.slider("Valitse ennustepäivien määrä:", 3, 10, 5)

# --- Uutisten haku ---
@st.cache_data
def fetch_news(symbol, api_key):
    url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-07-01&to=2024-07-26&token={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        news = response.json()
        return news[:5]
    else:
        return []

# --- Sentimenttianalyysi ---
def analyze_sentiment(news_list):
    sentiments = []
    for item in news_list:
        headline = item.get('headline', '')
        if headline:
            result = sentiment_pipeline(headline)[0]
            score = result['score']
            label = result['label']
            if label == 'NEGATIVE':
                sentiments.append(-score)
            else:
                sentiments.append(score)
    if sentiments:
        return sum(sentiments) / len(sentiments)
    return 0

# --- Historian ja featureiden haku ---
@st.cache_data
def load_data(symbol, days_ahead, sentiment_score):
    data = yf.download(symbol, start="2015-01-01", end="2024-01-01")
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['Return'] = data['Close'].pct_change()
    data = data.dropna()
    data['Sentiment'] = sentiment_score
    data['Target'] = np.where(data['Close'].shift(-days_ahead) > data['Close'], 1, 0)
    return data

# --- Uutisten näyttö ---
news = fetch_news(symbol, API_KEY)
st.subheader("🗞️ Viimeisimmät uutiset")
if news:
    for n in news:
        st.write(f"- {n['datetime'][:10]}: {n['headline']}")
else:
    st.write("Uutisia ei löytynyt tai API-virhe.")

# --- Sentimenttipiste ---
sentiment_score = analyze_sentiment(news)
st.write(f"Sentimenttipiste: {sentiment_score:.3f}")

# --- Ladataan osakedata ---
data = load_data(symbol, days_ahead, sentiment_score)
features = ['SMA_10', 'SMA_50', 'Return', 'Sentiment']
X = data[features]
y = data['Target']

# --- Mallin koulutus ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X[:-200], y[:-200])

# --- Ennuste uusimmalla datalla ---
latest = X.iloc[[-1]]
proba = model.predict_proba(latest)[0]
confidence = abs(proba[1] - proba[0])
prediction = model.predict(latest)[0]

# --- Riskitaso: luottamusraja ---
confidence_threshold = {
    "matala": 0.05,
    "keskitaso": 0.1,
    "korkea": 0.2
}[risk]

if confidence < confidence_threshold:
    suggestion = "🤔 PIDÄ (epävarma signaali)"
elif prediction == 1:
    suggestion = "📈 OSTA"
else:
    suggestion = "📉 MYY"

st.subheader("🔍 Agentin suositus")
st.write(f"**{suggestion}**")
st.write(f"Luottamus: `{confidence:.2f}`")

# --- Takautuva simulaatio (backtest) ---
st.subheader("🧪 Takautuva simulaatio")

capital = 10000
cash = capital
position = 0
portfolio_values = []
dates = []

for i in range(len(X) - 200, len(X) - days_ahead):
    row = X.iloc[[i]]
    pred = model.predict(row)[0]
    close_price = float(data['Close'].iloc[i])  # varmistetaan yksittäinen arvo

    if pred == 1 and cash >= close_price:
        position += 1
        cash -= close_price
    elif pred == 0 and position > 0:
        cash += close_price * position
        position = 0

    total_value = cash + position * close_price
    portfolio_values.append(total_value)
    dates.append(data.index[i])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(dates, portfolio_values, label="Agentin portfolio")
ax.set_title("Agentin tuotto (simuloitu)")
ax.set_ylabel("€")
ax.legend()
st.pyplot(fig)

final_return = portfolio_values[-1] - capital
st.markdown(f"**Lopputulos:** `{portfolio_values[-1]:.2f}€`")
st.markdown(f"**Voitto/Tappio:** `{final_return:+.2f}€`")
