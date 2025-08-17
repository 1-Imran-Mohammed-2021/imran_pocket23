# py_signal_bot.py (updated version)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
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
import threading
from threading import Thread
import websockets

# ==============================
# KEEP-ALIVE SERVER
# ==============================
try:
    from keep_alive import keep_alive
except ImportError:
    keep_alive = None
    logging.warning("keep_alive module not found or Flask not installed. Install with 'pip install flask' to enable keep_alive functionality.")

# ==============================
# CONFIGURATION
# ==============================
load_dotenv()
TELEGRAM_API_KEY = os.getenv('TELEGRAM_API_KEY')
TWELVE_DATA_API_KEY = os.getenv('TWELVE_DATA_API_KEY')
ADMIN_CHAT_ID = '5389240816'

# Validate API keys
if not TELEGRAM_API_KEY or TELEGRAM_API_KEY.strip() == "":
    raise ValueError("TELEGRAM_API_KEY is missing or empty in .env file. Please provide a valid API key.")
if not TWELVE_DATA_API_KEY or TWELVE_DATA_API_KEY.strip() == "":
    raise ValueError("TWELVE_DATA_API_KEY is missing or empty in .env file. Please provide a valid API key.")

# Generate or load encryption key
ENCRYPTION_KEY_FILE = 'encryption_key.key'
if not os.path.exists(ENCRYPTION_KEY_FILE):
    key = Fernet.generate_key()
    with open(ENCRYPTION_KEY_FILE, 'wb') as f:
        f.write(key)
    os.chmod(ENCRYPTION_KEY_FILE, 0o600)  # Restrict to owner read/write
with open(ENCRYPTION_KEY_FILE, 'rb') as f:
    ENCRYPTION_KEY = f.read()
cipher = Fernet(ENCRYPTION_KEY)

LOOKBACK = 58  # 1-hour history at 1-minute intervals
PRICE_THRESHOLD_BUY = 0.0001  # $0.0001 for buy signal
PRICE_THRESHOLD_SELL = -0.0001  # -$0.0001 for sell signal
MIN_PRICES_REQUIRED = LOOKBACK + 1
MAX_RETRIES = 3
FALLBACK_PRICE_COUNT = MIN_PRICES_REQUIRED * 2

# Rate limiting configuration
RATE_LIMIT_SECONDS = 30
MAX_REQUESTS_PER_WINDOW = 3
user_requests = defaultdict(list)

# Token configuration
TOKEN_EXPIRY_MINUTES = 10
tokens = {}

# Supported crypto pairs
crypto_symbols = {
    'BTC': {'pair': 'BTCUSDT', 'flag': '₿', 'coingecko_id': 'bitcoin'},
    'ETH': {'pair': 'ETHUSDT', 'flag': 'Ξ', 'coingecko_id': 'ethereum'},
    'BNB': {'pair': 'BNBUSDT', 'flag': '🟡', 'coingecko_id': 'binancecoin'},
    'SOL': {'pair': 'SOLUSDT', 'flag': '🌞', 'coingecko_id': 'solana'},
    'XRP': {'pair': 'XRPUSDT', 'flag': '⚪', 'coingecko_id': 'ripple'}
}

# Supported forex pairs
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

# Internationalization for messages
translations = {
    'ku': {
        'unauthorized': "🚫 تکایە پەیوەندی بەئادمینەوە بکە بۆ وەرگرتنی کلیل",
        'rate_limit': f"⚠️ تکایە جاوەرەنی بکە {RATE_LIMIT_SECONDS} دوای  هەوڵبدەوە.",
        'welcome': "بەخێربێت ! تکایە یەکێک لەمانەی خوارەوە هەڵبژێرە:",
        'select_crypto': "Select Crypto Pair:",
        'select_forex': "Select Forex Pair:",
        'select_interval': "تکایە ماوە هەڵبژێرە",
        'gathering_prices': "⏳ Gathering live prices for",
        'insufficient_data': "⚠️ Insufficient valid price data for",
        'insufficient_after_outlier': "⚠️ Insufficient valid price data after outlier removal for",
        'invalid_data_zero_std': "⚠️ Invalid price data for",
        'model_train_fail': "⚠️ Failed to train model for",
        'invalid_asset_type': "⚠️ تکایە ئەسێتێکی تەواوە هەڵبژێرە.",
        'invalid_pair': "⚠️ جووتێکی هەڵەت هەڵبژاردووە دووبارە هەوڵبدەوە .",
        'invalid_interval': "⚠️ Invalid interval selected.",
        'invalid_signal': "⚠️ Invalid signal request.",
        'error_processing': "⚠️ Error processing request. Please try again.",
        'error_pair_selection': "⚠️ Error processing pair selection. Please try again.",
        'error_sending': "⚠️ Error sending message. Please try again later.",
        'auth_usage': "⚠️ Usage: /auth <token>",
        'invalid_token_format': "⚠️ Invalid token format. Please provide a valid token.",
        'invalid_token': "⚠️ تۆکەنێی هەڵە تکایە پەیوەندی بە ئادمین بکە .",
        'authorized': "ئستا دەتوانی بەنجەبنێ بە /start بۆ دەست پێ کردن، بەسەرکەوتووی چاڵاک کرا !✅.",
        'new_user_authorized': "✅ New user authorized: [REDACTED]",
        'buy': 'کڕین',
        'sell': 'فرۆشتن',
        'hold': 'چاوەروان کردن',
        'admin_only': "🚫 تەنها ئادمین دەتوانێت کلیل درووست بکات.",
        'token_generated': f"🔑 New token: `{{token}}`\nExpires in {TOKEN_EXPIRY_MINUTES} minutes. Share with the user to authorize them."
    }
}

def get_message(key, lang='ku'):
    return translations.get(lang, translations['ku']).get(key, key)

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

# Single event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Redact sensitive data for logging
def redact_sensitive_data(data):
    if isinstance(data, dict):
        data = data.copy()
        for key in ['apiKey', 'token', 'key']:
            if key in data:
                data[key] = '[REDACTED]'
    return data

# ==============================
# AUTHORIZATION
# ==============================
def load_authorized_users():
    try:
        with open('authorized_users.json', 'rb') as f:
            encrypted_data = f.read()
            data = cipher.decrypt(encrypted_data)
            return json.loads(data.decode())
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.info(f"No authorized users found, creating empty list: {e}")
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

def cleanup_expired_tokens():
    now = datetime.now()
    expired = [token for token, (_, expiry) in tokens.items() if now > expiry]
    for token in expired:
        del tokens[token]
    logger.info(f"Cleaned up {len(expired)} expired tokens")

# Run cleanup every 5 minutes
def token_cleanup_thread():
    while True:
        cleanup_expired_tokens()
        time.sleep(300)  # 5 minutes

threading.Thread(target=token_cleanup_thread, daemon=True).start()

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
# BINANCE WEBSOCKET FOR CRYPTO PRICES
# ==============================
async def crypto_websocket():
    streams = "/".join([sym.lower() + "usdt@ticker" for sym in crypto_symbols])
    uri = f"wss://stream.binance.com:9443/stream?streams={streams}"
    while True:
        try:
            async with websockets.connect(uri) as ws:
                logger.info("Connected to Binance websocket")
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if 'data' in data:
                        d = data['data']
                        sym = d['s'][:-4].upper()  # BTCUSDT -> BTC
                        if sym in crypto_symbols:
                            price = float(d['c'])
                            price_cache[sym].append(price)
                            price_cache[sym] = price_cache[sym][- (LOOKBACK * 2):]
                            logger.debug(f"Updated price for {sym}: {price}")
        except Exception as e:
            logger.error(f"Binance websocket error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

# Start websocket in background
loop.create_task(crypto_websocket())

# ==============================
# PRICE FETCHING
# ==============================
async def fetch_live_price_async(symbol, asset_type, session, retries=MAX_RETRIES):
    if asset_type == 'crypto':
        # For crypto, use price_cache updated by websocket
        if price_cache[symbol]:
            return price_cache[symbol][-1]
        else:
            return None
    else:
        # Forex uses Twelve Data quote
        for attempt in range(retries):
            try:
                pair = get_supported_forex_pairs()[symbol]['id'].replace('/', '')
                url = f'https://api.twelvedata.com/quote?symbol={pair}&apikey={TWELVE_DATA_API_KEY}'
                async with session.get(url, timeout=5) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if 'close' not in data:
                        logger.error(f"Invalid response for {symbol}: {data}")
                        return None
                    price = float(data['close'])
                    price_cache[symbol].append(price)
                    price_cache[symbol] = price_cache[symbol][- (LOOKBACK * 2):]
                    logger.info(f"Fetched live price for {symbol}: {price}")
                    return price
            except (aiohttp.ClientError, ValueError, KeyError) as e:
                logger.error(f"Error fetching live price for {symbol} (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
    return None

async def collect_live_prices(symbol, asset_type, count):
    prices = []
    async with aiohttp.ClientSession() as session:
        for _ in range(count):
            price = await fetch_live_price_async(symbol, asset_type, session)
            if price and isinstance(price, (int, float)):
                prices.append(price)
            await asyncio.sleep(5 if asset_type == 'forex' else 1)
    return prices

async def fetch_historical_data(symbol, asset_type, session, retries=MAX_RETRIES):
    # Fetches 1 hour of 1min data (61 prices to include current)
    for attempt in range(retries):
        try:
            if asset_type == 'crypto':
                pair = crypto_symbols[symbol]['pair']
                url = f'https://api.binance.com/api/v3/klines?symbol={pair}&interval=1m&limit={MIN_PRICES_REQUIRED}'
                async with session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    data = await response.json()
                    prices = [float(item[4]) for item in data]  # close prices
                    logger.info(f"Crypto data for {symbol}: {prices[:5]}... (total {len(prices)})")
                    if len(prices) < MIN_PRICES_REQUIRED:
                        logger.warning(f"Insufficient historical data for {symbol}: {len(prices)} prices, needed {MIN_PRICES_REQUIRED}")
                    else:
                        logger.info(f"Fetched {len(prices)} historical prices from Binance for {symbol}")
                    return prices
            else:
                # Twelve Data for forex
                pair = get_supported_forex_pairs()[symbol]['id']
                url = f'https://api.twelvedata.com/time_series?symbol={pair}&interval=1min&outputsize={MIN_PRICES_REQUIRED}&apikey={TWELVE_DATA_API_KEY}'
                async with session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if 'values' not in data:
                        logger.error(f"Invalid response for {symbol}: {redact_sensitive_data(data)}")
                        return []
                    prices = [float(item['close']) for item in data['values']]
                    logger.info(f"Forex data for {symbol}: {prices[:5]}... (total {len(prices)})")
                    if len(prices) < MIN_PRICES_REQUIRED:
                        logger.warning(f"Insufficient historical data for {symbol}: {len(prices)} prices, needed {MIN_PRICES_REQUIRED}")
                    else:
                        logger.info(f"Fetched {len(prices)} historical prices from Twelve Data for {symbol}")
                    return prices
        except (aiohttp.ClientError, ValueError, KeyError) as e:
            logger.error(f"Error fetching historical data for {symbol} (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2)
            else:
                logger.error(f"Failed to fetch historical data for {symbol} after {retries} attempts")
    return []

# ==============================
# LSTM MODEL
# ==============================
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
        logger.warning(f"Failed to load model for {symbol}: {e}.")
        return None

# ==============================
# SIGNAL GENERATOR
# ==============================
def generate_signal(predicted, current, prices):
    price_diff = predicted - current
    recent_prices = np.array(prices[-20:])
    volatility = np.std(recent_prices / recent_prices.mean()) if len(recent_prices) > 1 else 0
    threshold_buy = PRICE_THRESHOLD_BUY * (1 + volatility)
    threshold_sell = PRICE_THRESHOLD_SELL * (1 + volatility)
    
    logger.info(f"Signal calc: predicted={predicted:.6f}, current={current:.6f}, price_diff={price_diff:.6f}, "
                f"volatility={volatility:.4f}, threshold_buy={threshold_buy:.6f}, threshold_sell={threshold_sell:.6f}")
    
    if price_diff > threshold_buy:
        return get_message('buy'), '🟢'
    elif price_diff < threshold_sell:
        return get_message('sell'), '🔴'
    else:
        return get_message('hold'), '⚪'

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
        f"{signal} ({last_price:.6f})",
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
            try:
                bot.send_photo(chat_id, photo)
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send photo to {chat_id}: {e}")
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
        try:
            bot.send_message(chat_id, get_message('unauthorized'))
        except telebot.apihelper.ApiTelegramException as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")
        logger.warning(f"Unauthorized access attempt by chat_id: [REDACTED]")
        return
    markup = InlineKeyboardMarkup()
    markup.row(
        InlineKeyboardButton('📊 Crypto', callback_data='type_crypto'),
        InlineKeyboardButton('💱 Forex', callback_data='type_forex')
    )
    try:
        bot.send_message(chat_id, get_message('welcome'), reply_markup=markup)
    except telebot.apihelper.ApiTelegramException as e:
        logger.error(f"Failed to send message to {chat_id}: {e}")
    logger.info(f"Authorized user [REDACTED] started bot")

@bot.message_handler(commands=['generate_token'])
def generate_token_command(message):
    chat_id = str(message.chat.id)
    if chat_id != ADMIN_CHAT_ID:
        try:
            bot.send_message(chat_id, get_message('admin_only'))
        except telebot.apihelper.ApiTelegramException as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")
        logger.warning(f"Non-admin [REDACTED] attempted to generate token")
        return
    token = generate_token(chat_id)
    try:
        bot.send_message(chat_id, get_message('token_generated').format(token=token), parse_mode='Markdown')
    except telebot.apihelper.ApiTelegramException as e:
        logger.error(f"Failed to send message to {chat_id}: {e}")
    logger.info(f"Admin [REDACTED] generated token")

@bot.message_handler(commands=['auth'])
def auth_command(message):
    chat_id = str(message.chat.id)
    parts = message.text.split()
    if len(parts) != 2:
        try:
            bot.send_message(chat_id, get_message('auth_usage'))
        except telebot.apihelper.ApiTelegramException as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")
        logger.warning(f"Invalid /auth command by [REDACTED]: incorrect number of arguments")
        return
    token = parts[1]
    try:
        uuid.UUID(token)  # Validate UUID format
    except ValueError:
        try:
            bot.send_message(chat_id, get_message('invalid_token_format'))
        except telebot.apihelper.ApiTelegramException as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")
        logger.warning(f"Invalid token format by [REDACTED]: {token}")
        return
    if validate_token(token, chat_id):
        try:
            bot.send_message(chat_id, get_message('authorized'))
            bot.send_message(ADMIN_CHAT_ID, get_message('new_user_authorized'))
        except telebot.apihelper.ApiTelegramException as e:
            logger.error(f"Failed to send message: {e}")
    else:
        try:
            bot.send_message(chat_id, get_message('invalid_token'))
        except telebot.apihelper.ApiTelegramException as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")
    logger.info(f"Authorization attempt by [REDACTED] with token: {token}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('type_'))
def choose_type(call):
    try:
        authorized_users = load_authorized_users()
        chat_id = str(call.message.chat.id)
        if chat_id not in authorized_users:
            try:
                bot.send_message(chat_id, get_message('unauthorized'))
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
            logger.warning(f"Unauthorized access attempt by chat_id: [REDACTED]")
            return
        asset_type = call.data.split('_')[1]
        if asset_type not in ['crypto', 'forex']:
            try:
                bot.send_message(chat_id, get_message('invalid_asset_type'))
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
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
        try:
            bot.send_message(call.message.chat.id, get_message(f'select_{asset_type}'), reply_markup=markup)
        except telebot.apihelper.ApiTelegramException as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")
        logger.info(f"Authorized user [REDACTED] chose {asset_type}")
    except Exception as e:
        try:
            bot.send_message(call.message.chat.id, get_message('error_processing'))
        except telebot.apihelper.ApiTelegramException as ex:
            logger.error(f"Failed to send message: {ex}")
        logger.error(f"Error in choose_type: {e}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('asset_'))
def choose_pair(call):
    try:
        authorized_users = load_authorized_users()
        chat_id = str(call.message.chat.id)
        if chat_id not in authorized_users:
            try:
                bot.send_message(chat_id, get_message('unauthorized'))
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
            logger.warning(f"Unauthorized access attempt by chat_id: [REDACTED]")
            return
        parts = call.data.split('_')
        if len(parts) != 3:
            try:
                bot.send_message(chat_id, get_message('invalid_pair'))
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
            logger.error(f"Invalid callback data: {call.data}")
            return
        _, asset_type, symbol = parts
        if asset_type not in ['crypto', 'forex'] or symbol not in (crypto_symbols if asset_type == 'crypto' else forex_symbols):
            try:
                bot.send_message(chat_id, get_message('invalid_pair'))
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
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
        try:
            bot.send_message(chat_id, f"{get_message('select_interval')} {flag} {symbol}:", reply_markup=markup)
        except telebot.apihelper.ApiTelegramException as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")
        logger.info(f"Authorized user [REDACTED] selected {symbol}")
    except Exception as e:
        try:
            bot.send_message(call.message.chat.id, get_message('error_pair_selection'))
        except telebot.apihelper.ApiTelegramException as ex:
            logger.error(f"Failed to send message: {ex}")
        logger.error(f"Error in choose_pair: {e}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('signal_'))
def handle_signal(call):
    try:
        authorized_users = load_authorized_users()
        chat_id = str(call.message.chat.id)
        if chat_id not in authorized_users:
            try:
                bot.send_message(chat_id, get_message('unauthorized'))
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
            logger.warning(f"Unauthorized access attempt by chat_id: [REDACTED]")
            return

        # Rate limiting
        now = datetime.now()
        user_requests[chat_id] = [t for t in user_requests[chat_id] if now - t < timedelta(seconds=RATE_LIMIT_SECONDS)]
        if len(user_requests[chat_id]) >= MAX_REQUESTS_PER_WINDOW:
            try:
                bot.send_message(chat_id, get_message('rate_limit'))
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
            logger.warning(f"Rate limit exceeded for user [REDACTED]")
            return
        user_requests[chat_id].append(now)

        parts = call.data.split('_')
        if len(parts) != 4:
            try:
                bot.send_message(chat_id, get_message('invalid_signal'))
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
            logger.error(f"Invalid callback data: {call.data}")
            return
        _, asset_type, symbol, interval = parts
        if asset_type not in ['crypto', 'forex'] or symbol not in (crypto_symbols if asset_type == 'crypto' else forex_symbols):
            try:
                bot.send_message(chat_id, get_message('invalid_pair'))
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
            logger.error(f"Invalid asset_type or symbol: {asset_type}, {symbol}")
            return
        try:
            interval = int(interval)
            if interval <= 0:
                raise ValueError("Interval must be positive")
        except ValueError:
            try:
                bot.send_message(chat_id, get_message('invalid_interval'))
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
            logger.error(f"Invalid interval: {interval}")
            return

        flag = crypto_symbols[symbol]['flag'] if asset_type == 'crypto' else forex_symbols[symbol]['flag']
        try:
            bot.send_message(chat_id, f"{get_message('gathering_prices')} {flag} {symbol} (1h history)...")
        except telebot.apihelper.ApiTelegramException as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")
        logger.info(f"Authorized user [REDACTED] requested signal for {symbol} ({interval}s)")

        # Fetch historical data
        async def get_historical_data():
            async with aiohttp.ClientSession() as session:
                return await fetch_historical_data(symbol, asset_type, session)

        historical_data_future = asyncio.run_coroutine_threadsafe(get_historical_data(), loop)
        try:
            prices = historical_data_future.result(timeout=10)
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching historical data for {symbol}")
            try:
                bot.send_message(chat_id, f"⚠️ Timeout fetching data for {flag} {symbol}. Please try again.")
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
            return

        if len(prices) == LOOKBACK:
            # If exactly LOOKBACK prices, try to append current price
            async def get_current_price():
                async with aiohttp.ClientSession() as session:
                    return await fetch_live_price_async(symbol, asset_type, session)

            current_price_future = asyncio.run_coroutine_threadsafe(get_current_price(), loop)
            try:
                current_price = current_price_future.result(timeout=10)
                if current_price and (not prices or current_price != prices[-1]):
                    prices.append(current_price)
                    price_cache[symbol].append(current_price)
            except asyncio.TimeoutError:
                logger.error(f"Timeout fetching current price for {symbol}")

        prices = [p for p in prices if p is not None]
        logger.info(f"Total prices collected for {symbol}: {len(prices)}")
        if len(prices) < MIN_PRICES_REQUIRED:
            # Fallback to collecting more live prices
            logger.warning(f"Insufficient data for {symbol}: {len(prices)} prices, fetching more...")
            async def collect_more_prices():
                async with aiohttp.ClientSession() as session:
                    return await collect_live_prices(symbol, asset_type, FALLBACK_PRICE_COUNT - len(prices))

            prices_future = asyncio.run_coroutine_threadsafe(collect_more_prices(), loop)
            try:
                new_prices = prices_future.result(timeout=60)
                prices.extend(new_prices)
                prices = prices[-MIN_PRICES_REQUIRED:]
            except asyncio.TimeoutError:
                logger.error(f"Timeout fetching additional prices for {symbol}")

        if len(prices) < MIN_PRICES_REQUIRED:
            markup = InlineKeyboardMarkup()
            markup.add(InlineKeyboardButton('🔄 Retry', callback_data=call.data))
            try:
                bot.send_message(chat_id, f"{get_message('insufficient_data')} {flag} {symbol}. Got {len(prices)} prices, need >={MIN_PRICES_REQUIRED}. Check connectivity.", reply_markup=markup)
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
            logger.warning(f"Insufficient data for {symbol}: {len(prices)} prices")
            return

        # Remove outliers
        prices_array = np.array(prices)
        std_dev = np.std(prices_array)
        if std_dev < 1e-6:
            logger.warning(f"Low standard deviation ({std_dev}) for {symbol}, skipping outlier removal")
            prices = prices_array.tolist()
        else:
            z_scores = np.abs((prices_array - np.mean(prices_array)) / std_dev)
            prices = prices_array[z_scores < 3].tolist()
            logger.info(f"After outlier removal, total prices for {symbol}: {len(prices)}")

        if len(prices) < MIN_PRICES_REQUIRED:
            markup = InlineKeyboardMarkup()
            markup.add(InlineKeyboardButton('🔄 Retry', callback_data=call.data))
            try:
                bot.send_message(chat_id, f"{get_message('insufficient_after_outlier')} {flag} {symbol}. Got {len(prices)} prices, need >={MIN_PRICES_REQUIRED}.", reply_markup=markup)
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
            logger.warning(f"Insufficient data after outlier removal for {symbol}: {len(prices)} prices")
            return

        # Generate signal
        scaler = MinMaxScaler()
        prices_scaled = scaler.fit_transform(np.array(prices).reshape(-1, 1))
        logger.info(f"prices_scaled shape: {prices_scaled.shape}")

        model = load_model(symbol)
        if model is None:
            try:
                bot.send_message(chat_id, f"⚠️ No pretrained model for {flag} {symbol}. Please contact admin to pretrain.")
            except telebot.apihelper.ApiTelegramException as e:
                logger.error(f"Failed to send message to {chat_id}: {e}")
            logger.error(f"No pretrained model for {symbol}")
            return

        input_data = torch.tensor(prices_scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1), dtype=torch.float32)
        logger.info(f"input_data shape for prediction: {input_data.shape}")
        model.eval()
        with torch.no_grad():
            pred_scaled = model(input_data).item()

        pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
        current_price = prices[-1]
        signal, signal_emoji = generate_signal(pred_price, current_price, prices)

        # Plot prices
        plot_prices(prices, symbol, signal, signal_emoji, chat_id)

        # Send signal message
        msg = f"{signal_emoji} <b>{flag} {symbol} Signal</b> (1h history)\n💰 Current: <b>{current_price:.6f}</b>\n📈 Predicted: <b>{pred_price:.6f}</b>\n📍 Action: <b>{signal}</b>"
        if std_dev < 1e-6:
            msg += "\n⚠️ Warning: Low price variability detected. Signal may be less reliable."
        try:
            bot.send_message(chat_id, msg, parse_mode='HTML')
        except telebot.apihelper.ApiTelegramException as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")
            try:
                bot.send_message(chat_id, get_message('error_sending'))
            except telebot.apihelper.ApiTelegramException as ex:
                logger.error(f"Failed to send fallback message to {chat_id}: {ex}")
        logger.info(f"Signal generated for {symbol}: {signal} (Current: {current_price:.6f}, Predicted: {pred_price:.6f})")
    except Exception as e:
        try:
            bot.send_message(call.message.chat.id, get_message('error_processing'))
        except telebot.apihelper.ApiTelegramException as ex:
            logger.error(f"Failed to send message: {ex}")
        logger.error(f"Error in handle_signal: {e}")

# ==============================
# UNIT TESTS
# ==============================
def run_tests():
    import unittest
    import tempfile
    from unittest.mock import patch

    class TestTradingBot(unittest.TestCase):
        @patch('aiohttp.ClientSession.get')
        async def test_fetch_historical_data(self, mock_get):
            mock_get.return_value.__aenter__.return_value.json.return_value = [[0, 0, 0, 0, 60000, 0]] * 61
            async with aiohttp.ClientSession() as session:
                prices = await fetch_historical_data('BTC', 'crypto', session)
            self.assertEqual(len(prices), 61)
            self.assertTrue(all(isinstance(p, float) for p in prices))

        @patch('__main__.bot.send_photo')
        def test_plot_prices(self, mock_send_photo):
            with tempfile.TemporaryDirectory() as tmpdir:
                prices = [60000 + i * 0.0001 for i in range(61)]
                symbol = 'TEST'
                signal = get_message('buy')
                signal_emoji = '🟢'
                chat_id = 12345
                plot_prices(prices, symbol, signal, signal_emoji, chat_id)
                self.assertFalse(os.path.exists(f"{symbol}_prices.png"), "Chart file should be deleted after generation")
                mock_send_photo.assert_called_once()

        def test_generate_signal(self):
            prices = [100, 100.0001, 100.0002, 100.0003, 100.0004] * 4  # 20 prices
            signal, emoji = generate_signal(predicted=100.0002, current=100.0000, prices=prices)
            self.assertEqual(signal, get_message('buy'))
            self.assertEqual(emoji, '🟢')

    suite = unittest.TestLoader().loadTestsFromTestCase(TestTradingBot)
    unittest.TextTestRunner().run(suite)

# ==============================
# START BOT
# ==============================
if __name__ == "__main__":
    # Initialize forex symbols
    forex_symbols = get_supported_forex_pairs()
    logger.info(f"Supported forex pairs: {list(forex_symbols.keys())}")

    # Start event loop in a separate thread
    def run_event_loop():
        loop.run_forever()

    threading.Thread(target=run_event_loop, daemon=True).start()

    logger.info("Starting trading bot...")
    run_tests()
    if keep_alive:
        keep_alive()  # Call from separate file
    else:
        logger.warning("keep_alive functionality disabled due to missing module or Flask")
    bot.infinity_polling(none_stop=True)