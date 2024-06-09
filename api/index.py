import os
import io
import string
import requests
import numpy as np
import pandas as pd
import httpx
import uvicorn
from dotenv import load_dotenv
from starlette.responses import JSONResponse
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.routing import Route
from starlette.exceptions import HTTPException as StarletteHTTPException

# Load environment variables
load_dotenv()

# Constants
BATCH_SIZE = 30

# Environment variables
api_url = os.getenv("API_URL")
json_url = os.getenv("JSON_URL")
bearer_token = os.getenv("BEARER_TOKEN")
dashboard_url = os.getenv("DASHBOARD_URL")

# Headers for API requests
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
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

async def categorize_transactions(df, bank_type):
    try:
        # Load category mappings
        response = requests.get(json_url, headers=headers, verify=False)
        response.raise_for_status()
        categories = response.json()

        results = []
        description_col, amount_col = column_heuristic(df, bank_type)

        for batch_start in range(0, len(df), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch_inputs = [
                f"{row[description_col]} ({row[amount_col]})"
                for _, row in df.iloc[batch_start:batch_end].iterrows()
            ]
            payload = {
                "inputs": batch_inputs,
                "parameters": {}
            }

            response = await query(payload)
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

# Process CSV files received from HTTP request
async def process_csv_files(request: Request):
    form = await request.form()
    files = form.getlist('files[]')
    processed_dataframes = []

    for file in files:
        try:
            bank_type, filename = file.filename.split('_', 1)
        except ValueError:
            return JSONResponse({"error": "Filename format is incorrect. Expecting <bank_type>_<filename>"}, status_code=400)

        acceptable_file_types = ['text/csv', 'application/vnd.ms-excel']
        if file.content_type not in acceptable_file_types:
            return JSONResponse({"error": f"File {filename} is not a CSV file"}, status_code=400)

        content = await file.read()
        try:
            if bank_type == "Costco":
                df = pd.read_csv(io.BytesIO(content), delimiter=',', skiprows=5)
            else:
                df = pd.read_csv(io.BytesIO(content), delimiter=',')

            categorized_df = await categorize_transactions(df, bank_type)
            categorized_df = replace_non_compliant_values(categorized_df)
            processed_dataframes.append(categorized_df.to_dict('records'))
        except Exception as e:
            return JSONResponse({"error": f"Failed to process {filename}: {e}"}, status_code=500)

    payload = {
        "data": processed_dataframes,
        "reviewId": request.query_params.get("reviewId")
    }

    try:
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(dashboard_url, headers=headers2, json=payload, timeout=None)
            response.raise_for_status()
            return JSONResponse({"message": "JSON data sent successfully"})
    except httpx.HTTPStatusError as e:
        return JSONResponse({"error": f"Failed to send JSON data: {e}"}, status_code=response.status_code)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Determine columns to use for description and amount, and clean unwanted columns
def column_heuristic(df, bank_type):
    df.rename(columns={col: col.strip(string.punctuation + string.whitespace) for col in df.columns}, inplace=True)

    if bank_type == "Chase":
        try:
            df.drop(columns=["Post Date", "Category", "Type", "Memo"], inplace=True)
            df['Amount'] = df['Amount'].replace(r'[^\d.-]', '', regex=True).astype(float)
        except KeyError as e:
            raise ValueError(f"Missing expected column: {e}")
    
    elif bank_type == "Costco":
        try:
            df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0).apply(lambda x: -abs(x))
            df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0).apply(lambda x: abs(x))
            df['Amount'] = df['Debit'] + df['Credit']
            df.drop(columns=["Debit", "Credit", "Category"], inplace=True)
        except KeyError as e:
            raise ValueError(f"Missing expected column: {e}")

    elif bank_type == "Oriental Bank":
        try:
            df.loc[df['<Additional Info>'].notna(), '<Description>'] = df['<Additional Info>']
            df['<Withdrawal Amount>'] = df['<Withdrawal Amount>'].fillna(0).apply(lambda x: -abs(x))
            df['<Deposit Amount>'] = df['<Deposit Amount>'].fillna(0).apply(lambda x: abs(x))
            df['<Amount>'] = df['<Withdrawal Amount>'] + df['<Deposit Amount>']
            df.drop(columns=["<Withdrawal Amount>", "<Deposit Amount>", "<CheckNum>", "<Additional Info>"], inplace=True)
            df['Amount'] = df['<Amount>'].replace(r'[^\d.-]', '', regex=True).astype(float)
        except KeyError as e:
            raise ValueError(f"Missing expected column: {e}")
    
    elif bank_type == "Penfed":
        try:
            df.drop(columns=["Card Number Last 4", "Posted Date", "Transaction Type"], inplace=True)
            df['Amount'] = df['Amount'].replace(r'[^\d.-]', '', regex=True).astype(float)
        except KeyError as e:
            raise ValueError(f"Missing expected column: {e}")

    elif bank_type in ["Visa", "Banco Popular"]:
        required_columns = ['Date', 'Description', 'Amount']
        if not all(column in df.columns for column in required_columns):
            raise ValueError("The columns are not present in the DataFrame")
        df['Amount'] = df['Amount'].replace(r'[^\d.-]', '', regex=True).astype(float)

    description_col = next((col for col in df.columns if 'description' in col.lower()), None)
    amount_col = next((col for col in df.columns if 'amount' in col.lower()), None)

    if description_col is None or amount_col is None:
        raise ValueError("Required columns for description or amount not found")

    return description_col, amount_col

# Replace non-compliant values
def replace_non_compliant_values(df):
    df.replace([np.inf, -np.inf, np.nan], [None, None, None], inplace=True)
    return df

# Application setup
app = Starlette(
    debug=True,
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
    routes=[
        Route("/", endpoint=process_csv_files, methods=["POST"]),
    ]
)

# Custom exception handler
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
