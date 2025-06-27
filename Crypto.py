# -----------------------------------------
# COMPLETE CODE WITH TIME SERIES DROPDOWN
# -----------------------------------------

import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 1: Fetch top 50 cryptocurrencies
def get_top_50_cryptos():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 50,
        'page': 1,
        'sparkline': False
    }
    response = requests.get(url, params=params).json()
    return pd.DataFrame(response)[['id', 'symbol', 'name', 'market_cap', 'current_price']]

crypto_df = get_top_50_cryptos()

# Step 2: Expanded industry map
industry_map = {
    "bitcoin": "Digital Gold",
    "litecoin": "Payments",
    "dogecoin": "Payments",
    "ripple": "Cross-border Payments",
    "ethereum": "Smart Contracts / Web3",
    "cardano": "Smart Contracts / Web3",
    "solana": "Smart Contracts / Web3",
    "polygon": "Scalability",
    "avalanche-2": "Smart Contracts / Web3",
    "polkadot": "Interoperability",
    "tron": "Entertainment / Web3",
    "algorand": "Smart Contracts / Web3",
    "tezos": "Smart Contracts / Web3",
    "aave": "DeFi",
    "uniswap": "DeFi",
    "maker": "DeFi",
    "dai": "Stablecoins / DeFi",
    "chainlink": "Data Oracles",
    "the-graph": "Data Oracles",
    "filecoin": "Decentralized Storage",
    "arweave": "Decentralized Storage",
    "monero": "Privacy",
    "mina-protocol": "Zero Knowledge / Privacy",
    "vechain": "Supply Chain",
    "hedera-hashgraph": "Enterprise / Web3",
    "flow": "NFT / Web3",
    "decentraland": "Metaverse",
    "sandbox": "Metaverse",
    "theta-token": "Video Streaming",
    "render-token": "GPU / AI",
    "crypto-com-chain": "Exchange Token",
    "okb": "Exchange Token",
    "leo-token": "Exchange Token",
}

crypto_df['industry'] = crypto_df['id'].map(industry_map).fillna("Other")

# Step 3: Define broader buckets
bucket_map = {
    "Smart Contracts / Web3": "Smart Contracts & Web3",
    "Scalability": "Infrastructure / Scaling",
    "Interoperability": "Infrastructure / Scaling",
    "Cross-border Payments": "Payments",
    "Digital Gold": "Payments",
    "Payments": "Payments",
    "DeFi": "DeFi",
    "Stablecoins / DeFi": "DeFi",
    "Privacy": "Privacy",
    "Zero Knowledge / Privacy": "Privacy",
    "Data Oracles": "Storage & Data",
    "Decentralized Storage": "Storage & Data",
    "Supply Chain": "Enterprise / Utility",
    "Enterprise / Web3": "Enterprise / Utility",
    "NFT / Web3": "Smart Contracts & Web3",
    "Metaverse": "Smart Contracts & Web3",
    "GPU / AI": "Enterprise / Utility",
    "Exchange Token": "Enterprise / Utility",
    "Video Streaming": "Smart Contracts & Web3",
    "Other": "Other"
}
crypto_df['industry_bucket'] = crypto_df['industry'].map(bucket_map).fillna("Other")

print("Industry distribution:")
print(crypto_df['industry'].value_counts())
print("\nBucket distribution:")
print(crypto_df['industry_bucket'].value_counts())

# Step 4: Generate mock scores
def generate_mock_scores(df):
    df['adoption_score'] = [random.uniform(1, 10) for _ in range(len(df))]
    df['developer_score'] = [random.uniform(1, 10) for _ in range(len(df))]
    df['partnership_score'] = [random.uniform(1, 10) for _ in range(len(df))]
    df['total_score'] = df[['adoption_score','developer_score','partnership_score']].mean(axis=1)
    return df

crypto_df = generate_mock_scores(crypto_df)

# Step 5: Plot top 10 by total score
plt.figure(figsize=(12,6))
sns.barplot(
    data=crypto_df.sort_values("total_score",ascending=False).head(10),
    x='name', y='total_score', hue='industry_bucket'
)
plt.xticks(rotation=45)
plt.title("Top 10 Cryptos by Total Score & Industry Bucket")
plt.tight_layout()
plt.show()

# Step 6: Recommend top N by industry
def recommend_by_industry(df, selected_industry, top_n=5):
    return df[df['industry']==selected_industry].sort_values("total_score",ascending=False).head(top_n)

print("\nTop Smart Contracts / Web3 tokens:")
print(recommend_by_industry(crypto_df, "Smart Contracts / Web3")[['name','total_score']])

# Step 7: Mock classification & confusion matrix
crypto_df['predicted_bucket'] = [random.choice(crypto_df['industry_bucket'].unique()) for _ in range(len(crypto_df))]
cm = confusion_matrix(
    crypto_df['industry_bucket'], 
    crypto_df['predicted_bucket'], 
    labels=crypto_df['industry_bucket'].unique()
)
labels = crypto_df['industry_bucket'].unique()

plt.figure(figsize=(8,6))
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot(xticks_rotation=45, cmap='Purples', colorbar=True)
plt.title("Confusion Matrix: Industry Buckets")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# TIME SERIES ANALYSIS (log-scale subplots with dropdown)
# ------------------------------------------------

# dropdown simulation in terminal
options = {
    '1': 14,
    '2': 21,
    '3': 28
}
print("\nSelect time series range for top-5 cryptos:")
print("1: 2 weeks")
print("2: 3 weeks")
print("3: 4 weeks")
print("Press Enter to default to 10 weeks.")

choice = input("Enter 1, 2, or 3 (default 10 weeks): ").strip()

if choice in options:
    days = options[choice]
else:
    days = 70  # default to 10 weeks

print(f"\nFetching time series for {days} days...")

def get_historical_prices(coin_id, days=days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': days}
    data = requests.get(url, params=params).json()
    if 'prices' in data:
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        prices['date'] = pd.to_datetime(prices['timestamp'], unit='ms')
        prices.set_index('date', inplace=True)
        return prices[['price']]
    else:
        return pd.DataFrame()

top5 = crypto_df.sort_values("market_cap", ascending=False).head(5)

fig, axs = plt.subplots(len(top5), 1, figsize=(12, 14), sharex=True)

for ax, coin in zip(axs, top5['id']):
    ts_df = get_historical_prices(coin, days=days)
    if not ts_df.empty:
        ax.plot(ts_df.index, ts_df['price'], marker='o', label=coin.capitalize())
        ax.set_title(f"{coin.capitalize()} Price (Log Scale, {days} days)")
        ax.set_ylabel("Price (USD)")
        ax.set_yscale("log")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()

axs[-1].set_xlabel("Date")
plt.tight_layout()
plt.show()
