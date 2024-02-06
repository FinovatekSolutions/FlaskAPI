from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello from Python! :D 🐍'

@app.route('/about')
def about():
    return 'About'