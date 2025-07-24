# AAI-Project

# 🪙 Crypto Recommender App

**Crypto Recommender** is an interactive Streamlit web app that helps users identify suitable cryptocurrency investments based on their risk profile. It integrates real-time data from the CoinGecko API, visualizations, machine learning mockups, and even sends personalized coin reports via email in PDF format.

---

## 🔍 Features

- 🧠 **Risk Profiling Questionnaire**: Tailors recommendations based on user investment behavior.
- 📈 **Top 50 Cryptocurrencies**: Pulled in real-time using CoinGecko API.
- 🧩 **Scoring System**: Cryptos scored on `Adoption`, `Developer Activity`, and `Partnership Strength`.
- 📊 **Visual Insights**:
  - 30-day price trend chart.
  - Confusion matrix (mock-up for illustrative ML classification).
- 📨 **Email Reporting**:
  - Generates and emails a personalized PDF report using `reportlab`.
  - Includes pricing data, total score, and chart.
