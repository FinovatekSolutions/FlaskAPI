from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify(message='Hello from Python! :D 🐍')

@app.route('/about')
def about():
    return jsonify(message='about')