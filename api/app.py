import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['xls', 'csv', 'pdf', 'xlsx', 'xml'])
app = Flask(__name__)
CORS(app, origins=["https://finovatek.vercel.app/", "http://localhost:3000"])

def allowedFile(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/upload', methods=['POST'])
def fileUpload():
    if request.method == 'POST':
        file = request.files.getlist('file')
        print(request.files, "....")
        for f in file:
            print(f.filename)
            filename = secure_filename(f.filename)
            print(allowedFile(filename))
            if allowedFile(filename):
                df = pd.read_csv(f)
                print(df)
            else:
                return jsonify({'message': 'File type not allowed'}), 400
        return jsonify({"name": filename, "status": "success"})
    else:
        return jsonify({"status": "Upload API GET Request Running"})

if __name__ == '__main__':
    app.run(debug=True)