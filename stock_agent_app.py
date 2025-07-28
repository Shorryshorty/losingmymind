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
