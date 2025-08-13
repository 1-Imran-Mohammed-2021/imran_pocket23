import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from collections import defaultdict
import logging

import os
from dotenv import load_dotenv
import aiohttp
import asyncio
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import time
import requests
import shutil
from cryptography.fernet import Fernet
from datetime import datetime, timedelta
import uuid

# ==============================
# CONFIGURATION
# ==============================
load_dotenv()
TELEGRAM_API_KEY='8224411567:AAF8QFxlN2SYZeUadqzt3C-xoF8fwwgQ7b0'
TWELVE_DATA_API_KEY='66b7861c69a44a9f8177e5ea3313e0d6'
ADMIN_CHAT_ID='5389240816'
CURRENCYFREAKS_API_KEY='b9fe0f84cb944521888236f316b8f3a9'
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')


# Validate CurrencyFreaks API key
if not CURRENCYFREAKS_API_KEY or CURRENCYFREAKS_API_KEY.strip() == "":
    raise ValueError("CURRENCYFREAKS_API_KEY is missing or empty in .env file. Please provide a valid API key.")

# Generate or load encryption key
ENCRYPTION_KEY_FILE = 'encryption_key.key'
if not os.path.exists(ENCRYPTION_KEY_FILE):
    key = Fernet.generate_key()
    with open(ENCRYPTION_KEY_FILE, 'wb') as f:
        f.write(key)
with open(ENCRYPTION_KEY_FILE, 'rb') as f:
    ENCRYPTION_KEY = f.read()
cipher = Fernet(ENCRYPTION_KEY)

LOOKBACK = 10
BASE_THRESHOLD_BUY = 0.002
BASE_THRESHOLD_SELL = -0.002
MIN_PRICES_REQUIRED = LOOKBACK + 1
MAX_RETRIES = 3
FALLBACK_PRICE_COUNT = MIN_PRICES_REQUIRED * 2

# Rate limiting configuration
RATE_LIMIT_SECONDS = 30
user_requests = defaultdict(list)

# Token configuration
TOKEN_EXPIRY_MINUTES = 10
tokens = {}

# Supported crypto pairs with flags
crypto_symbols = {
    'BTC': {'pair': 'BTCUSDT', 'flag': '₿'},
    'ETH': {'pair': 'ETHUSDT', 'flag': 'Ξ'},
    'BNB': {'pair': 'BNBUSDT', 'flag': '🟡'},
    'SOL': {'pair': 'SOLUSDT', 'flag': '🌞'},
    'XRP': {'pair': 'XRPUSDT', 'flag': '⚪'}
}

# Supported forex pairs
def get_supported_forex_pairs():
    return {
        'EURUSD': {'id': 'EUR/USD', 'flag': '🇪🇺🇺🇸'},
        'USDJPY': {'id': 'USD/JPY', 'flag': '🇺🇸🇯🇵'},
        'GBPUSD': {'id': 'GBP/USD', 'flag': '🇬🇧🇺🇸'},
        'MADUSD': {'id': 'MAD/USD', 'flag': '🇲🇦🇺🇸'},
        'SARCNY': {'id': 'SAR/CNY', 'flag': '🇸🇦🇨🇳'},
        'CADCHF': {'id': 'CAD/CHF', 'flag': '🇨🇦🇨🇭'},
        'QARCNY': {'id': 'QAR/CNY', 'flag': '🇶🇦🇨🇳'},
        'USDCOP': {'id': 'USD/COP', 'flag': '🇺🇸🇨🇴'},
        'USDTHB': {'id': 'USD/THB', 'flag': '🇺🇸🇹🇭'},
        'GBPAUD': {'id': 'GBP/AUD', 'flag': '🇬🇧🇦🇺'},
        'UAHUSD': {'id': 'UAH/USD', 'flag': '🇺🇦🇺🇸'},
        'USDSGD': {'id': 'USD/SGD', 'flag': '🇺🇸🇸🇬'},
        'NZDUSD': {'id': 'NZD/USD', 'flag': '🇳🇿🇺🇸'},
        'USDDZD': {'id': 'USD/DZD', 'flag': '🇺🇸🇩🇿'},
        'USDPKR': {'id': 'USD/PKR', 'flag': '🇺🇸🇵🇰'},
        'USDINR': {'id': 'USD/INR', 'flag': '🇺🇸🇮🇳'},
        'EURRUB': {'id': 'EUR/RUB', 'flag': '🇪🇺🇷🇺'},
        'GBPJPY': {'id': 'GBP/JPY', 'flag': '🇬🇧🇯🇵'},
        'EURGBP': {'id': 'EUR/GBP', 'flag': '🇪🇺🇬🇧'},
        'USDVND': {'id': 'USD/VND', 'flag': '🇺🇸🇻🇳'},
        'USDRUB': {'id': 'USD/RUB', 'flag': '🇺🇸🇷🇺'},
        'NGNUSD': {'id': 'NGN/USD', 'flag': '🇳🇬🇺🇸'},
        'EURCHF': {'id': 'EUR/CHF', 'flag': '🇪🇺🇨🇭'},
        'CHFNOK': {'id': 'CHF/NOK', 'flag': '🇨🇭🇳🇴'},
        'BHDCNY': {'id': 'BHD/CNY', 'flag': '🇧🇭🇨🇳'},
        'JODCNY': {'id': 'JOD/CNY', 'flag': '🇯🇴🇨🇳'},
        'NZDJPY': {'id': 'NZD/JPY', 'flag': '🇳🇿🇯🇵'},
        'USDBDT': {'id': 'USD/BDT', 'flag': '🇺🇸🇧🇩'},
        'USDMXN': {'id': 'USD/MXN', 'flag': '🇺🇸🇲🇽'},
        'EURNZD': {'id': 'EUR/NZD', 'flag': '🇪🇺🇳🇿'},
        'USDCNH': {'id': 'USD/CNH', 'flag': '🇺🇸🇨🇳'},
        'YERUSD': {'id': 'YER/USD', 'flag': '🇾🇪🇺🇸'},
        'USDPHP': {'id': 'USD/PHP', 'flag': '🇺🇸🇵🇭'},
        'EURHUF': {'id': 'EUR/HUF', 'flag': '🇪🇺🇭🇺'},
        'USDEGP': {'id': 'USD/EGP', 'flag': '🇺🇸🇪🇬'},
        'CHFJPY': {'id': 'CHF/JPY', 'flag': '🇨🇭🇯🇵'},
        'ZARUSD': {'id': 'ZAR/USD', 'flag': '🇿🇦🇺🇸'},
        'LBPUSD': {'id': 'LBP/USD', 'flag': '🇱🇧🇺🇸'},
        'USDARS': {'id': 'USD/ARS', 'flag': '🇺🇸🇦🇷'},
        'AEDCNY': {'id': 'AED/CNY', 'flag': '🇦🇪🇨🇳'},
        'CADJPY': {'id': 'CAD/JPY', 'flag': '🇨🇦🇯🇵'},
        'USDBRL': {'id': 'USD/BRL', 'flag': '🇺🇸🇧🇷'},
        'AUDCHF': {'id': 'AUD/CHF', 'flag': '🇦🇺🇨🇭'},
        'EURJPY': {'id': 'EUR/JPY', 'flag': '🇪🇺🇯🇵'},
        'EURTRY': {'id': 'EUR/TRY', 'flag': '🇪🇺🇹🇷'},
        'KESUSD': {'id': 'KES/USD', 'flag': '🇰🇪🇺🇸'},
        'OMRCNY': {'id': 'OMR/CNY', 'flag': '🇴🇲🇨🇳'},
        'TNDUSD': {'id': 'TND/USD', 'flag': '🇹🇳🇺🇸'},
        'USDCAD': {'id': 'USD/CAD', 'flag': '🇺🇸🇨🇦'},
        'USDCHF': {'id': 'USD/CHF', 'flag': '🇺🇸🇨🇭'},
        'USDCLP': {'id': 'USD/CLP', 'flag': '🇺🇸🇨🇱'},
        'USDIDR': {'id': 'USD/IDR', 'flag': '🇺🇸🇮🇩'},
        'USDMYR': {'id': 'USD/MYR', 'flag': '🇺🇸🇲🇾'}
    }

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

bot = telebot.TeleBot(TELEGRAM_API_KEY)
price_cache = defaultdict(list)

# Initialize forex_symbols
forex_symbols = {}

# ==============================
# AUTHORIZATION
# ==============================
def load_authorized_users():
    try:
        with open('authorized_users.json', 'rb') as f:
            encrypted_data = f.read()
            data = cipher.decrypt(encrypted_data)
            return json.loads(data.decode())
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        logger.info("No authorized users found, creating empty list")
        return []

def save_authorized_users(users):
    os.makedirs('auth', exist_ok=True)
    encrypted_data = cipher.encrypt(json.dumps(users).encode())
    with open('authorized_users.json', 'wb') as f:
        f.write(encrypted_data)
    logger.info("Saved authorized users")

# ==============================
# TOKEN MANAGEMENT
# ==============================
def generate_token(admin_chat_id):
    token = str(uuid.uuid4())
    expiry = datetime.now() + timedelta(minutes=TOKEN_EXPIRY_MINUTES)
    tokens[token] = (admin_chat_id, expiry)
    logger.info(f"Generated token for admin [REDACTED]")
    return token

def validate_token(token, user_chat_id):
    if token not in tokens:
        logger.warning(f"Invalid token attempt: {token}")
        return False
    admin_chat_id, expiry = tokens[token]
    if datetime.now() > expiry:
        logger.warning(f"Expired token attempt: {token}")
        del tokens[token]
        return False
    authorized_users = load_authorized_users()
    if str(user_chat_id) in authorized_users:
        logger.info(f"User [REDACTED] already authorized")
        del tokens[token]
        return False
    authorized_users.append(str(user_chat_id))
    save_authorized_users(authorized_users)
    del tokens[token]
    logger.info(f"Authorized new user [REDACTED] with token {token}")
    return True

# ==============================
# PRICE CACHING
# ==============================
def save_prices(symbol, prices):
    os.makedirs('price_data', exist_ok=True)
    with open(f'price_data/{symbol}.json', 'w') as f:
        json.dump(prices, f)
    logger.info(f"Saved {len(prices)} prices for {symbol}")

def load_prices(symbol):
    try:
        with open(f'price_data/{symbol}.json', 'r') as f:
            prices = json.load(f)
        logger.info(f"Loaded {len(prices)} cached prices for {symbol}")
        return prices
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        logger.info(f"No cached prices for {symbol}")
        return []

# ==============================
# PRICE FETCHING
# ==============================
async def fetch_live_price_async(symbol, asset_type, session, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            if asset_type == 'crypto':
                url = f'https://api.binance.com/api/v3/avgPrice?symbol={crypto_symbols[symbol]["pair"]}'
                headers = {'X-MBX-APIKEY': BINANCE_API_KEY} if BINANCE_API_KEY else {}
                async with session.get(url, headers=headers, timeout=5) as response:
                    response.raise_for_status()
                    data = await response.json()
                    price = float(data['price']) if 'price' in data else None
                    if price:
                        price_cache[symbol].append(price)
                        price_cache[symbol] = price_cache[symbol][-100:]
                        logger.info(f"Fetched live price for {symbol}: {price}")
                    return price
            else:
                url = f'https://api.currencyfreaks.com/v2.0/rates/latest?apikey={CURRENCYFREAKS_API_KEY}'
                async with session.get(url, timeout=5) as response:
                    response.raise_for_status()
                    data = await response.json()
                    logger.debug(f"CurrencyFreaks API response for {symbol}: {data}")
                    if 'rates' not in data:
                        logger.error(f"Invalid response for {symbol}: {data}")
                        return None
                    rates = data['rates']
                    rates['USD'] = '1.0'  # Base is USD
                    base, quote = forex_symbols[symbol]['id'].split('/')
                    if base not in rates or quote not in rates:
                        logger.error(f"Unsupported currencies for {symbol}: {base} or {quote}")
                        return None
                    price = float(rates[quote]) / float(rates[base])
                    price_cache[symbol].append(price)
                    price_cache[symbol] = price_cache[symbol][-100:]
                    logger.info(f"Fetched live price for {symbol}: {price}")
                    return price
        except (aiohttp.ClientError, ValueError, KeyError) as e:
            logger.error(f"Error fetching live price for {symbol} (attempt {attempt+1}/{retries}): {e}")
            if 'response' in locals() and response:
                logger.error("API response: [REDACTED]")
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
            await asyncio.sleep(3)
    return prices

def fetch_historical_data(symbol, asset_type, interval='1m', limit=100, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            if asset_type == 'crypto':
                url = f'https://api.binance.com/api/v3/klines?symbol={crypto_symbols[symbol]["pair"]}&interval={interval}&limit={limit}'
                headers = {'X-MBX-APIKEY': BINANCE_API_KEY} if BINANCE_API_KEY else {}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                prices = [float(item[4]) for item in data]
                if len(prices) < MIN_PRICES_REQUIRED:
                    logger.warning(f"Insufficient historical data for {symbol}: {len(prices)} prices, needed {MIN_PRICES_REQUIRED}")
                else:
                    logger.info(f"Fetched {len(prices)} historical prices from Binance for {symbol}")
                return prices
            else:
                return []
        except (requests.RequestException, ValueError) as e:
            logger.error(f"Error fetching historical data for {symbol} (attempt {attempt+1}/{retries}): {e}")
            if 'response' in locals():
                logger.error("API response: [REDACTED]")
            if attempt < retries - 1:
                time.sleep(2)
            continue
    logger.error(f"Failed to fetch historical data for {symbol} after {retries} attempts")
    return []

# ==============================
# LSTM MODEL
# ==============================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        h_0 = torch.zeros(2 * 3, x.size(0), 128).to(x.device)
        c_0 = torch.zeros(2 * 3, x.size(0), 128).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        return self.fc(out[:, -1, :])

def save_model(model, symbol):
    os.makedirs('models', exist_ok=True)
    state_dict = model.state_dict()
    encrypted_data = cipher.encrypt(json.dumps({k: v.tolist() for k, v in state_dict.items()}).encode())
    with open(f'models/{symbol}.pth', 'wb') as f:
        f.write(encrypted_data)
    logger.info(f"Saved model for {symbol}")

def load_model(symbol):
    model = LSTMModel()
    try:
        with open(f'models/{symbol}.pth', 'rb') as f:
            encrypted_data = f.read()
            data = cipher.decrypt(encrypted_data)
            state_dict = json.loads(data.decode())
            state_dict = {k: torch.tensor(v) for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        logger.info(f"Loaded model for {symbol}")
        return model
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to load model for {symbol}: {e}. Training new model.")
        return None

# ==============================
# TRAINING FUNCTION
# ==============================
def train_model(prices_scaled, max_epochs=50, patience=5):
    prices_scaled = np.array(prices_scaled).reshape(-1, 1)
    logger.info(f"prices_scaled shape in train_model: {prices_scaled.shape}")

    X, y = [], []
    for i in range(len(prices_scaled) - LOOKBACK):
        X.append(prices_scaled[i:i+LOOKBACK].flatten())
        y.append(prices_scaled[i+LOOKBACK].item())
    if len(X) < 1:
        logger.error(f"Not enough data to train model: {len(prices_scaled)} prices, need >{LOOKBACK}")
        return None

    X, y = np.array(X), np.array(y)
    logger.info(f"X shape after np.array: {X.shape}")
    logger.info(f"y shape: {y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
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

# ==============================
# SIGNAL GENERATOR
# ==============================
def generate_signal(predicted, current, prices):
    pct_change = (predicted - current) / current
    recent_prices = np.array(prices[-20:])
    volatility = np.std(recent_prices / recent_prices.mean()) if len(recent_prices) > 1 else 0
    threshold_buy = BASE_THRESHOLD_BUY * (1 + volatility)
    threshold_sell = BASE_THRESHOLD_SELL * (1 + volatility)
    
    if pct_change > threshold_buy:
        return 'کڕین', '🟢'
    elif pct_change < threshold_sell:
        return 'فرۆشتن', '🔴'
    else:
        return 'چاوەروان کردن', '⚪'

# ==============================
# VISUALIZATION
# ==============================
def plot_prices(prices, symbol, signal, signal_emoji, chat_id):
    plt.figure(figsize=(10, 5))
    plt.plot(prices, label=f"{symbol} Prices", color='blue')
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"{symbol} Price Data with Signal")
    plt.legend()
    plt.grid(True)
    
    last_price = prices[-1]
    last_index = len(prices) - 1
    plt.scatter([last_index], [last_price], color='red', s=100, label='Latest Price')
    plt.annotate(
        f"{signal} ({last_price:.4f})",
        xy=(last_index, last_price),
        xytext=(last_index, last_price + (max(prices) - min(prices)) * 0.05),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=10,
        color='black',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
    )
    
    plt.savefig(f"{symbol}_prices.png")
    plt.close()
    if chat_id:
        with open(f"{symbol}_prices.png", 'rb') as photo:
            bot.send_photo(chat_id, photo)
    os.remove(f"{symbol}_prices.png")
    logger.info(f"Sent price chart for {symbol} with signal: {signal}")

# ==============================
# TELEGRAM BOT HANDLERS
# ==============================
@bot.message_handler(commands=['start'])
def start(message):
    authorized_users = load_authorized_users()
    chat_id = str(message.chat.id)
    if chat_id not in authorized_users:
        bot.send_message(chat_id, "🚫 تکایە پەیوەندی بەئادمینەوە بکە بۆ وەرگرتنی کلیل ")
        logger.warning(f"Unauthorized access attempt by chat_id: [REDACTED]")
        return
    markup = InlineKeyboardMarkup()
    markup.row(
        InlineKeyboardButton('📊 Crypto', callback_data='type_crypto'),
        InlineKeyboardButton('💱 Forex', callback_data='type_forex')
    )
    bot.send_message(chat_id, "بەخێربێت ! تکایە یەکێک لەمانەی خوارەوە هەڵبژێرە:", reply_markup=markup)
    logger.info(f"Authorized user [REDACTED] started bot")

@bot.message_handler(commands=['generate_token'])
def generate_token_command(message):
    chat_id = str(message.chat.id)
    if chat_id != ADMIN_CHAT_ID:
        bot.send_message(chat_id, "🚫 تەنها ئادمین دەتوانێت کلیل درووست بکات.")
        logger.warning(f"Non-admin [REDACTED] attempted to generate token")
        return
    token = generate_token(chat_id)
    bot.send_message(chat_id, f"🔑 New token: `{token}`\nExpires in {TOKEN_EXPIRY_MINUTES} minutes. Share with the user to authorize them.", parse_mode='Markdown')
    logger.info(f"Admin [REDACTED] generated token")

@bot.message_handler(commands=['auth'])
def auth_command(message):
    chat_id = str(message.chat.id)
    try:
        token = message.text.split()[1]
    except IndexError:
        bot.send_message(chat_id, "⚠️ Usage: /auth <token>")
        logger.warning(f"Invalid /auth command by [REDACTED]: no token provided")
        return
    if validate_token(token, chat_id):
        bot.send_message(chat_id, "ئستا دەتوانی بەنجەبنێ بە /start بۆ دەست پێ کردن، بەسەرکەوتووی چاڵاک کرا !✅.")
        bot.send_message(ADMIN_CHAT_ID, f"✅ New user authorized: [REDACTED]")
    else:
        bot.send_message(chat_id, "⚠️ تۆکەنێی هەڵە تکایە پەیوەندی بە ئادمین بکە .")
    logger.info(f"Authorization attempt by [REDACTED] with token: {token}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('type_'))
def choose_type(call):
    try:
        authorized_users = load_authorized_users()
        chat_id = str(call.message.chat.id)
        if chat_id not in authorized_users:
            bot.send_message(chat_id, "🚫 تکایە پەیوەندی بە ئادمینەوە بکە بۆ وەرگرتنی کلیی چونەژورەوە")
            logger.warning(f"Unauthorized access attempt by chat_id: [REDACTED]")
            return
        asset_type = call.data.split('_')[1]
        if asset_type not in ['crypto', 'forex']:
            bot.send_message(chat_id, "⚠️ تکایە ئەسێتێکی تەواوە هەڵبژێرە.")
            logger.error(f"Invalid asset type: {asset_type}")
            return
        markup = InlineKeyboardMarkup()
        symbols = crypto_symbols if asset_type == 'crypto' else forex_symbols
        sorted_symbols = sorted(symbols.keys())
        for i in range(0, len(sorted_symbols), 3):
            row_buttons = [
                InlineKeyboardButton(f"{symbols[sym]['flag']} {sym}", callback_data=f'asset_{asset_type}_{sym}')
                for sym in sorted_symbols[i:i+3]
            ]
            markup.row(*row_buttons)
        bot.send_message(call.message.chat.id, f"Select {'Crypto' if asset_type == 'crypto' else 'Forex'} Pair:", reply_markup=markup)
        logger.info(f"Authorized user [REDACTED] chose {asset_type}")
    except Exception as e:
        bot.send_message(call.message.chat.id, "⚠️ Error processing request. Please try again.")
        logger.error(f"Error in choose_type: {e}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('asset_'))
def choose_pair(call):
    try:
        authorized_users = load_authorized_users()
        chat_id = str(call.message.chat.id)
        if chat_id not in authorized_users:
            bot.send_message(chat_id, "🚫تکایە پەیوەندی بە ئادمینەوە بکە بۆ وەرگرتنی کیلیل")
            logger.warning(f"Unauthorized access attempt by chat_id: [REDACTED]")
            return
        parts = call.data.split('_')
        if len(parts) != 3:
            bot.send_message(chat_id, "⚠️ جووتێکی هەڵەت هەڵبژاردووە دووبارە هەوڵبدەوە .")
            logger.error(f"Invalid callback data: {call.data}")
            return
        _, asset_type, symbol = parts
        if asset_type not in ['crypto', 'forex'] or symbol not in (crypto_symbols if asset_type == 'crypto' else forex_symbols):
            bot.send_message(chat_id, "⚠️ Invalid pair selected.")
            logger.error(f"Invalid asset_type or symbol: {asset_type}, {symbol}")
            return
        flag = crypto_symbols[symbol]['flag'] if asset_type == 'crypto' else forex_symbols[symbol]['flag']
        markup = InlineKeyboardMarkup()
        markup.row(
            InlineKeyboardButton('⏱️ 5s Signal', callback_data=f'signal_{asset_type}_{symbol}_5'),
            InlineKeyboardButton('⏱️ 10s Signal', callback_data=f'signal_{asset_type}_{symbol}_10')
        )
        markup.row(
            InlineKeyboardButton('⏱️ 60s Signal', callback_data=f'signal_{asset_type}_{symbol}_60'),
            InlineKeyboardButton('⏱️ 90s Signal', callback_data=f'signal_{asset_type}_{symbol}_90')
        )
        bot.send_message(chat_id, f"تکایە ماوە هەڵبژێرە {flag} {symbol}:", reply_markup=markup)
        logger.info(f"Authorized user [REDACTED] selected {symbol}")
    except Exception as e:
        bot.send_message(call.message.chat.id, "⚠️ Error processing pair selection. Please try again.")
        logger.error(f"Error in choose_pair: {e}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('signal_'))
def handle_signal(call):
    try:
        authorized_users = load_authorized_users()
        chat_id = str(call.message.chat.id)
        if chat_id not in authorized_users:
            bot.send_message(chat_id, "🚫 Unauthorized access. Please contact the admin for an authorization token.")
            logger.warning(f"Unauthorized access attempt by chat_id: [REDACTED]")
            return

        # Rate limiting
        now = datetime.now()
        user_requests[chat_id] = [t for t in user_requests[chat_id] if now - t < timedelta(seconds=RATE_LIMIT_SECONDS)]
        if len(user_requests[chat_id]) >= 1:
            bot.send_message(chat_id, f"⚠️ تکایە جاوەرەنی بکە {RATE_LIMIT_SECONDS} دوای  هەوڵبدەوە.")
            logger.warning(f"Rate limit exceeded for user [REDACTED]")
            return
        user_requests[chat_id].append(now)

        parts = call.data.split('_')
        if len(parts) != 4:
            bot.send_message(chat_id, "⚠️ Invalid signal request.")
            logger.error(f"Invalid callback data: {call.data}")
            return
        _, asset_type, symbol, interval = parts
        if asset_type not in ['crypto', 'forex'] or symbol not in (crypto_symbols if asset_type == 'crypto' else forex_symbols):
            bot.send_message(chat_id, "⚠️ Invalid pair selected.")
            logger.error(f"Invalid asset_type or symbol: {asset_type}, {symbol}")
            return
        try:
            interval = int(interval)
            if interval <= 0:
                raise ValueError("Interval must be positive")
        except ValueError:
            bot.send_message(chat_id, "⚠️ Invalid interval selected.")
            logger.error(f"Invalid interval: {interval}")
            return

        flag = crypto_symbols[symbol]['flag'] if asset_type == 'crypto' else forex_symbols[symbol]['flag']
        bot.send_message(chat_id, f"⏳ Gathering live prices for {flag} {symbol} ({interval}s)...")
        logger.info(f"Authorized user [REDACTED] requested signal for {symbol} ({interval}s)")

        prices = price_cache[symbol]
        if len(prices) < MIN_PRICES_REQUIRED:
            prices = load_prices(symbol)
            logger.info(f"Initial price cache for {symbol}: {len(prices)} prices")
        if len(prices) < MIN_PRICES_REQUIRED:
            logger.info(f"Price cache has {len(prices)} prices for {symbol}, fetching historical data")
            prices = fetch_historical_data(symbol, asset_type, limit=max(100, MIN_PRICES_REQUIRED * 2))
            if len(prices) >= MIN_PRICES_REQUIRED:
                save_prices(symbol, prices)
            if len(prices) < MIN_PRICES_REQUIRED:
                logger.warning(f"Failed to fetch enough historical data for {symbol}: {len(prices)} prices")
                logger.info(f"Collecting {FALLBACK_PRICE_COUNT} live prices for {symbol}")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    new_prices = loop.run_until_complete(collect_live_prices(symbol, asset_type, FALLBACK_PRICE_COUNT))
                    for price in new_prices:
                        if price and isinstance(price, (int, float)):
                            prices.append(price)
                            price_cache[symbol].append(price)
                            price_cache[symbol] = price_cache[symbol][-100:]
                finally:
                    loop.close()

        remaining_prices_needed = max(0, MIN_PRICES_REQUIRED - len(prices))
        if remaining_prices_needed > interval:
            logger.info(f"Extending live price collection to {remaining_prices_needed} seconds for {symbol}")
            interval = remaining_prices_needed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            new_prices = loop.run_until_complete(collect_live_prices(symbol, asset_type, interval))
            for price in new_prices:
                if price and isinstance(price, (int, float)):
                    prices.append(price)
                    price_cache[symbol].append(price)
                    price_cache[symbol] = price_cache[symbol][-100:]
        finally:
            loop.close()

        prices = [p for p in prices if p is not None]
        logger.info(f"Total prices collected for {symbol}: {len(prices)}")
        if len(prices) < MIN_PRICES_REQUIRED:
            markup = InlineKeyboardMarkup()
            markup.add(InlineKeyboardButton('🔄 Retry', callback_data=call.data))
            bot.send_message(chat_id, f"⚠️ Insufficient valid price data for {flag} {symbol}. Got {len(prices)} prices, need >={MIN_PRICES_REQUIRED}. Check API key or connectivity.", reply_markup=markup)
            logger.warning(f"Insufficient data for {symbol}: {len(prices)} prices")
            return

        # Remove outliers, but skip if standard deviation is zero
        prices_array = np.array(prices)
        std_dev = np.std(prices_array)
        if std_dev > 0:
            z_scores = np.abs((prices_array - np.mean(prices_array)) / std_dev)
            prices = prices_array[z_scores < 3].tolist()
            logger.info(f"After outlier removal, total prices for {symbol}: {len(prices)}")
        else:
            logger.warning(f"Standard deviation is zero for {symbol}, skipping outlier removal")
            prices = prices_array.tolist()

        if len(prices) < MIN_PRICES_REQUIRED:
            markup = InlineKeyboardMarkup()
            markup.add(InlineKeyboardButton('🔄 Retry', callback_data=call.data))
            bot.send_message(chat_id, f"⚠️ Insufficient valid price data after outlier removal for {flag} {symbol}. Got {len(prices)} prices, need >={MIN_PRICES_REQUIRED}.", reply_markup=markup)
            logger.warning(f"Insufficient data after outlier removal for {symbol}: {len(prices)} prices")
            return

        # Train model and generate signal
        scaler = MinMaxScaler()
        prices_scaled = scaler.fit_transform(np.array(prices).reshape(-1, 1))
        logger.info(f"prices_scaled shape: {prices_scaled.shape}")

        model = load_model(symbol)
        if model is None:
            logger.info(f"No pretrained model for {symbol}, training new model")
            model = train_model(prices_scaled)
            if model is None:
                bot.send_message(chat_id, f"⚠️ Failed to train model for {flag} {symbol}. Please try again.")
                logger.error(f"Model training failed for {symbol}")
                return
            save_model(model, symbol)

        input_data = torch.tensor(prices_scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1), dtype=torch.float32)
        logger.info(f"input_data shape for prediction: {input_data.shape}")
        model.eval()
        with torch.no_grad():
            pred_scaled = model(input_data).item()

        pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
        current_price = prices[-1]
        signal, signal_emoji = generate_signal(pred_price, current_price, prices)

        # Plot prices with signal annotation
        plot_prices(prices, symbol, signal, signal_emoji, chat_id)

        # Send signal message
        msg = f"{signal_emoji} <b>{flag} {symbol} Signal</b> ({interval}s)\n💰 Current: <b>{current_price:.4f}</b>\n📈 Predicted: <b>{pred_price:.4f}</b>\n📍 Action: <b>{signal}</b>"
        bot.send_message(chat_id, msg, parse_mode='HTML')
        logger.info(f"Signal generated for {symbol}: {signal} (Current: {current_price:.4f}, Predicted: {pred_price:.4f})")
    except Exception as e:
        bot.send_message(call.message.chat.id, "⚠️ Error processing signal request. Please try again.")
        logger.error(f"Error in handle_signal: {e}")

# ==============================
# UNIT TESTS
# ==============================
def run_tests():
    import unittest
    import tempfile
    from unittest.mock import patch

    class TestTradingBot(unittest.TestCase):
        def test_fetch_historical_data(self):
            try:
                prices = fetch_historical_data('BTC', 'crypto', limit=50)
                self.assertGreaterEqual(len(prices), MIN_PRICES_REQUIRED, "Should fetch at least 21 prices")
            except Exception as e:
                logger.warning(f"Test fetch_historical_data failed: {e}")
                self.skipTest(f"API unavailable: {e}")

        def test_train_model(self):
            prices = [60000 + i * 10 for i in range(50)]
            scaler = MinMaxScaler()
            prices_scaled = scaler.fit_transform(np.array(prices).reshape(-1, 1))
            model = train_model(prices_scaled)
            self.assertIsNotNone(model, "Model training should succeed")

        @patch('__main__.bot.send_photo')
        def test_plot_prices(self, mock_send_photo):
            with tempfile.TemporaryDirectory() as tmpdir:
                prices = [60000 + i * 10 for i in range(50)]
                symbol = 'TEST'
                signal = 'BUY'
                signal_emoji = '🟢'
                chat_id = 12345
                plot_prices(prices, symbol, signal, signal_emoji, chat_id)
                self.assertFalse(os.path.exists(f"{symbol}_prices.png"), "Chart file should be deleted after generation")
                mock_send_photo.assert_called_once()

    suite = unittest.TestLoader().loadTestsFromTestCase(TestTradingBot)
    unittest.TextTestRunner().run(suite)

# ==============================
# START BOT
# ==============================
if __name__ == "__main__":
    # Initialize forex symbols
    forex_symbols = get_supported_forex_pairs()
    logger.info(f"Supported forex pairs: {list(forex_symbols.keys())}")

    # Clear existing models
    if os.path.exists('models'):
        shutil.rmtree('models')
        logger.info("Cleared existing model files to ensure compatibility")
    
    logger.info("Starting trading bot...")
    run_tests()
    bot.infinity_polling(none_stop=True)
