import os
from dotenv import load_dotenv
load_dotenv()

import json
from cryptography.fernet import Fernet
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import aiohttp
import asyncio
import requests
import logging

# Setup logging for training script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CONFIG
LOOKBACK = 10
MIN_PRICES_REQUIRED = LOOKBACK + 1
MAX_RETRIES = 3
FALLBACK_PRICE_COUNT = MIN_PRICES_REQUIRED * 2
CURRENCYFREAKS_API_KEY = os.getenv('CURRENCYFREAKS_API_KEY')

crypto_symbols = {
    'BTC': {'pair': 'BTCUSDT', 'flag': '₿', 'coingecko_id': 'bitcoin'},
    'ETH': {'pair': 'ETHUSDT', 'flag': 'Ξ', 'coingecko_id': 'ethereum'},
    'BNB': {'pair': 'BNBUSDT', 'flag': '🟡', 'coingecko_id': 'binancecoin'},
    'SOL': {'pair': 'SOLUSDT', 'flag': '🌞', 'coingecko_id': 'solana'},
    'XRP': {'pair': 'XRPUSDT', 'flag': '⚪', 'coingecko_id': 'ripple'}
}

def get_supported_forex_pairs():
    return {
        'EURUSD': {'id': 'EUR/USD', 'flag': '🇪🇺🇺🇸'},
        'USDJPY': {'id': 'USD/JPY', 'flag': '🇺🇸🇯🇵'},
        'GBPUSD': {'id': 'GBP/USD', 'flag': '🇬🇧🇺🇸'},
        'AUDUSD': {'id': 'AUD/USD', 'flag': '🇦🇺🇺🇸'},
        'USDCAD': {'id': 'USD/CAD', 'flag': '🇺🇸🇨🇦'},
        'USDCHF': {'id': 'USD/CHF', 'flag': '🇺🇸🇨🇭'},
        'NZDUSD': {'id': 'NZD/USD', 'flag': '🇳🇿🇺🇸'},
        'EURJPY': {'id': 'EUR/JPY', 'flag': '🇪🇺🇯🇵'},
        'GBPJPY': {'id': 'GBP/JPY', 'flag': '🇬🇧🇯🇵'},
        'EURGBP': {'id': 'EUR/GBP', 'flag': '🇪🇺🇬🇧'}
    }

# Encryption key loading
ENCRYPTION_KEY_FILE = 'encryption_key.key'
if not os.path.exists(ENCRYPTION_KEY_FILE):
    key = Fernet.generate_key()
    with open(ENCRYPTION_KEY_FILE, 'wb') as f:
        f.write(key)
    os.chmod(ENCRYPTION_KEY_FILE, 0o600)
with open(ENCRYPTION_KEY_FILE, 'rb') as f:
    ENCRYPTION_KEY = f.read()
cipher = Fernet(ENCRYPTION_KEY)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        return self.fc(out[:, -1, :])

async def fetch_live_price_async(symbol, asset_type, session, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            if asset_type == 'crypto':
                coingecko_id = crypto_symbols[symbol]["coingecko_id"]
                url = 'https://api.coingecko.com/api/v3/simple/price'
                params = {'ids': coingecko_id, 'vs_currencies': 'usd'}
                async with session.get(url, params=params, timeout=5) as response:
                    response.raise_for_status()
                    data = await response.json()
                    price = float(data[coingecko_id]['usd']) if coingecko_id in data else None
                    if price:
                        logger.info(f"Fetched live price for {symbol}: {price}")
                    return price
            else:
                url = f'https://api.currencyfreaks.com/v2.0/rates/latest?apikey={CURRENCYFREAKS_API_KEY}'
                async with session.get(url, timeout=5) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if 'rates' not in data:
                        logger.error(f"Invalid response for {symbol}: {data}")
                        return None
                    rates = data['rates']
                    rates['USD'] = '1.0'
                    base, quote = forex_symbols[symbol]['id'].split('/')
                    if base not in rates or quote not in rates:
                        logger.error(f"Unsupported currencies for {symbol}: {base} or {quote}")
                        return None
                    try:
                        price = float(rates[quote]) / float(rates[base])
                    except (ValueError, ZeroDivisionError) as e:
                        logger.error(f"Error calculating price for {symbol}: {e}")
                        return None
                    logger.info(f"Fetched live price for {symbol}: {price}")
                    return price
        except Exception as e:
            logger.error(f"Error fetching live price for {symbol} (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2)
            continue
    logger.error(f"Failed to fetch live price for {symbol} after {retries} attempts")
    return None

async def collect_live_prices(symbol, asset_type, count):
    prices = []
    async with aiohttp.ClientSession() as session:
        for _ in range(count):
            price = await fetch_live_price_async(symbol, asset_type, session)
            if price and isinstance(price, (int, float)):
                prices.append(price)
            await asyncio.sleep(2)
    return prices

def fetch_historical_data(symbol, asset_type, days=1, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            if asset_type == 'crypto':
                coingecko_id = crypto_symbols[symbol]["coingecko_id"]
                url = f'https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart'
                params = {'vs_currency': 'usd', 'days': days}
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                prices = [item[1] for item in data.get('prices', [])]
                if len(prices) < MIN_PRICES_REQUIRED:
                    logger.warning(f"Insufficient historical data for {symbol}: {len(prices)} prices, needed {MIN_PRICES_REQUIRED}")
                else:
                    logger.info(f"Fetched {len(prices)} historical prices from CoinGecko for {symbol}")
                return prices
            else:
                return []
        except (requests.RequestException, ValueError) as e:
            logger.error(f"Error fetching historical data for {symbol} (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
            continue
    logger.error(f"Failed to fetch historical data for {symbol} after {retries} attempts")
    return []

def train_model(prices_scaled, max_epochs=50, patience=5):
    prices_scaled = np.array(prices_scaled).reshape(-1, 1)
    logger.info(f"prices_scaled shape in train_model: {prices_scaled.shape}")

    X, y = [], []
    for i in range(len(prices_scaled) - LOOKBACK):
        X.append(prices_scaled[i:i+LOOKBACK])
        y.append(prices_scaled[i+LOOKBACK].item())
    if len(X) < 1:
        logger.error(f"Not enough data to train model: {len(prices_scaled)} prices, need >{LOOKBACK}")
        return None

    X, y = np.array(X), np.array(y)
    logger.info(f"X shape after np.array: {X.shape}")
    logger.info(f"y shape: {y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")
    
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train.squeeze())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output.squeeze(), y_val.squeeze())
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Trained model with final validation loss: {best_loss:.6f}")
    return model

def save_model(model, symbol):
    os.makedirs('models', exist_ok=True)
    state_dict = model.state_dict()
    encrypted_data = cipher.encrypt(json.dumps({k: v.tolist() for k, v in state_dict.items()}).encode())
    with open(f'models/{symbol}.pth', 'wb') as f:
        f.write(encrypted_data)
    logger.info(f"Saved model for {symbol}")

# Main training logic
if __name__ == "__main__":
    forex_symbols = get_supported_forex_pairs()
    symbols = list(crypto_symbols.keys()) + list(forex_symbols.keys())
    for symbol in symbols:
        asset_type = 'crypto' if symbol in crypto_symbols else 'forex'
        prices = fetch_historical_data(symbol, asset_type, days=7)  # Increased to 7 days for better training
        if len(prices) < MIN_PRICES_REQUIRED:
            logger.info(f"Collecting fallback prices for {symbol}")
            prices = asyncio.run(collect_live_prices(symbol, asset_type, FALLBACK_PRICE_COUNT))
        if len(prices) >= MIN_PRICES_REQUIRED:
            scaler = MinMaxScaler()
            prices_scaled = scaler.fit_transform(np.array(prices).reshape(-1, 1))
            model = train_model(prices_scaled)
            if model:
                save_model(model, symbol)
        else:
            logger.error(f"Failed to collect enough data for {symbol}")