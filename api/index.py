import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
import requests
import string
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app, origins=["*"])

api_url = os.getenv("API_URL")
json_url = os.getenv("JSON_URL")
bearer_token = os.getenv("BEARER_TOKEN")
dashboard_url = os.getenv("DASHBOARD_URL")

BATCH_SIZE = 30  # Adjust based on experimentation

headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {bearer_token}",
    "Content-Type": "application/json"
}

headers2 = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}


async def query(payload):
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            print("Failed to decode JSON from response:", response.text)
            return None
    else:
        print("Received a non-200 response:", response.status_code)
        return None

@app.route("/", methods=["POST"])
def process_csv_files():
    files = request.files.getlist('files[]')
    processed_dataframes = []

    for file in files:
        bank_type, filename = file.filename.split('_')[0], file.filename.split('_')[1]
        print(f"Processing file: {file}")
        print("File", file)
        print("Bank Type", bank_type)
        print("File Name", filename)
        print("File Type", file.content_type)
        acceptable_file_types = ['text/csv', 'application/vnd.ms-excel']

        if file.mimetype not in acceptable_file_types:
            print(f"File {filename} is not a CSV file.")
            return jsonify({"error": f"File {filename} is not a CSV file"}), 500

        content = file.read()
        try:
            if bank_type == "Costco":
                df = pd.read_csv(io.BytesIO(content), delimiter=',', skiprows=5)
            else:
                df = pd.read_csv(io.BytesIO(content), delimiter=',')

            categorized_df = categorize_transactions(df, bank_type)
            categorized_df = replace_non_compliant_values(categorized_df)
            processed_dataframes.append(categorized_df.to_dict('records'))
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            return jsonify({"error": f"Failed to process {filename}"}), 500

    target_endpoint = dashboard_url
    reviewId = request.args.get("reviewId")
    payload = {"data": processed_dataframes, "reviewId": reviewId}
    print(payload)

    try:
        response = requests.post(target_endpoint, headers=headers2, json=payload)
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Content: {response.text}")
        if response.status_code == 200:
            return jsonify({"message": "JSON data sent successfully"})
        else:
            return jsonify({"error": "Failed to send JSON data"}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def categorize_transactions(df, bank_type):
    try:
        response = requests.get(json_url, headers=headers)
        categories = response.json()

        results = []
        batch_inputs = []
        description_col, amount_col = column_heuristic(df, bank_type)

        for batch_start in range(0, len(df), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch_inputs = [
                str(row[description_col]) + " (" + str(row[amount_col]) + ")" for _, row in
                df.iloc[batch_start:batch_end].iterrows()
            ]
            payload = {
                "inputs": batch_inputs,
                "parameters": {}
            }

            response = query(payload)
            if response is None or not isinstance(response, list):
                response = [{'label': 'Unknown_'}] * len(batch_inputs)

            for resp in response:
                number = resp.get('label', 'Unknown_').split('_')[1]

                if resp.get("score") > 0.80:
                    category = categories.get(number, 'Unknown Category')
                else:
                    category = "No Category"
                results.append(category)

        df['Category'] = results
        return df
    except Exception as e:
        print(f"Error during transaction categorization: {e}")
        df['Category'] = ['Error Processing'] * len(df)
        return df


def column_heuristic(df, bank_type):
    description_col = None
    amount_col = None

    df.rename(columns={col: col.strip(string.punctuation + string.whitespace) for col in df.columns}, inplace=True)

    if bank_type == "Chase":
        try:
            del df["Post Date"], df["Category"], df["Type"], df["Memo"]
            df['Amount'] = df['Amount'].replace(r'[^\d.-]', '', regex=True).astype(float)
        except Exception as e:
            print(f"Error finding the column {e} in the DataFrame")
            return jsonify({"error": f"The files you uploaded are not accepted"}), 500
   
    if bank_type == "Costco":
        try:
            # Convert 'Debit' and 'Credit' to numeric, set errors='coerce' to handle non-numeric values by setting them to NaN
            df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce')
            df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce')
            # print(df.head())
            # Replace NaN values with 0 for calculation
            df['Debit'] = df['Debit'].fillna(0)
            df['Credit'] = df['Credit'].fillna(0)

            # Ensure correct signs: withdrawals should be negative, deposits should be positive
            df['Debit'] = df['Debit'].apply(lambda x: -abs(x) if x != 0 else 0)
            df['Credit'] = df['Credit'].apply(lambda x: abs(x) if x != 0 else 0)

            # Create the 'Amount' column by summing the two columns
            df['Amount'] = df['Debit'] + df['Credit']
            del df['Debit'], df['Credit'], df["Category"]
            df['Amount'] = df['Amount'].replace(r'[^\d.-]', '', regex=True).astype(float)
        except Exception as e:
            print(f"Error finding the column {e} in the DataFrame")
            return jsonify({"error": f"The files you uploaded are not accepted"}, status_code=500)
    if bank_type == "Oriental Bank":
        try:
            # Replace the value in Description with the one in Additional Info column if not null
            df.loc[df['<Additional Info>'].notna(), '<Description>'] = df['<Additional Info>']

            # Replace NaN values with 0 for calculation
            df['<Withdrawal Amount>'] = df['<Withdrawal Amount>'].fillna(0)
            df['<Deposit Amount>'] = df['<Deposit Amount>'].fillna(0)

            # Ensure correct signs: withdrawals should be negative, deposits should be positive
            df['<Withdrawal Amount>'] = df['<Withdrawal Amount>'].apply(lambda x: -abs(x) if x != 0 else 0)
            df['<Deposit Amount>'] = df['<Deposit Amount>'].apply(lambda x: abs(x) if x != 0 else 0)

            # Create the 'Amount' column by summing the two columns
            df['<Amount>'] = df['<Withdrawal Amount>'] + df['<Deposit Amount>']
            del df['<Withdrawal Amount>'], df['<Deposit Amount>'], df["<CheckNum>"], df['<Additional Info>']
            df['Amount'] = df['Amount'].replace(r'[^\d.-]', '', regex=True).astype(float)
        except Exception as e:
            print(f"Error finding the column {e} in the DataFrame")
            return jsonify({"error": f"The files you uploaded are not accepted"}, status_code=500)
    if bank_type == "Penfed":
        try:
            del df["Card Number Last 4"], df["Posted Date"], df["Transaction Type"]
            df['Amount'] = df['Amount'].replace(r'[^\d.-]', '', regex=True).astype(float)
        except Exception as e:
            print(f"Error finding the column {e} in the DataFrame")
            return jsonify({"error": f"The files you uploaded are not accepted"}, status_code=500)
    if bank_type == "Visa" or bank_type == "Banco Popular":
        try:
            columns_to_check = ['Date', 'Description', 'Amount']
            are_columns_present = all(column in df.columns for column in columns_to_check)
            if not are_columns_present:
                raise ValueError("The columns are not present in the DataFrame")
            df['Amount'] = df['Amount'].replace(r'[^\d.-]', '', regex=True).astype(float)
        except Exception as e:
            print(f"{e}")
            return jsonify({"error": f"The files you uploaded are not accepted"}, status_code=500)

    for col in df.columns:
        if (col.find("Description") != -1) or ('description' in col.lower()):
            description_col = col
        elif (col.find("Amount") != -1) or ('amount' in col.lower()):
            amount_col = col

    return description_col, amount_col

def replace_non_compliant_values(df):
    df.replace([np.inf, -np.inf, np.nan], [None, None, None], inplace=True)
    return df

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)