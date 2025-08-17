# keep_alive.py (new separate file)

from flask import Flask
import logging
from threading import Thread

logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    return "Bot is alive!"

def run(port=8080, retries=3):
    for attempt in range(retries):
        try:
            app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
            logger.info(f"Flask server running on port {port}")
            return
        except OSError as e:
            if 'Address already in use' in str(e) and attempt < retries - 1:
                logger.warning(f"Port {port} in use, trying port {port + 1}")
                port += 1
                continue
            logger.error(f"Failed to start Flask server: {e}")
            raise

def keep_alive():
    try:
        t = Thread(target=run, kwargs={'port': 8080}, daemon=True)
        t.start()
        logger.info("Started keep_alive Flask server in separate thread")
    except Exception as e:
        logger.error(f"Failed to start keep_alive thread: {e}")