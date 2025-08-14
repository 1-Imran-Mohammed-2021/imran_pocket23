from flask import Flask
from threading import Thread

app = Flask(__name__)

@app.route('/')
def index():
    return "Alive"

def run():
    # Runs the Flask server on port 8080
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    # Starts the server in a separate thread
    t = Thread(target=run)
    t.start()
