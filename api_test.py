from starlette.responses import JSONResponse
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.routing import Route
import numpy as np
import pandas as pd
import io  # For converting bytes to a format pandas can read
import requests
import uvicorn
import httpx
import string

# Determine an optimal batch size
BATCH_SIZE = 30  # Adjust based on experimentation

# Your existing setup for API interaction
API_URL = "https://thwcvf276mq5w9os.us-east-1.aws.endpoints.huggingface.cloud"

# URL where categories JSON object is stored
json_url = 'https://huggingface.co/Finovatek/Categorization-Model/raw/main/label_to_category_mapping.json'

headers = {
    "Accept": "application/json",
    "Authorization": "Bearer {Add authentication token to use}",
    "Content-Type": "application/json" 
}

async def query(payload):
    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            try:
                return response.json()  # Ensure response is JSON and awaitable
            except ValueError:  # Includes JSONDecodeError
                print("Failed to decode JSON from response:", response.text)
                return None  # Or handle as appropriate
        else:
            print("Received a non-200 response:", response.status_code)
            return None  # Or handle as appropriate



async def categorize_transactions(df, bank_type):
    try:
        # Loading category mappings
        response = requests.get(json_url, headers=headers)
        categories = response.json()

        results = []
        batch_inputs = []
        description_col, amount_col = column_heuristic(df, bank_type)

        for batch_start in range(0, len(df), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch_inputs = [
                str(row[description_col]) + " (" + str(row[amount_col]) + ")" for _, row in df.iloc[batch_start:batch_end].iterrows()
            ]
            payload = {
                "inputs": batch_inputs,
                "parameters": {}
            }
            
            # Send the batch request
            response = await query(payload)
            if response is None or not isinstance(response, list):
                response = [{'label': 'Unknown_'}] * len(batch_inputs)  # Fallback response

            for resp in response:
                number = resp.get('label', 'Unknown_').split('_')[1]
                category = categories.get(number, 'Unknown Category')
                results.append(category)
    
        df['Category'] = results
        return df
    except Exception as e: 
        print(f"Error during transaction categorization: {e}")
        df['Category'] = ['Error Processing'] * len(df)  # Default error category
        return df

# Function that receives HTTP request with form data 
async def process_csv_files(request: Request):
    form = await request.form()
    files = form.getlist('files[]') 
    processed_dataframes = []
    
    for file in files:
        filename, bank_type = file.filename.split('_')[0], file.filename.split('_')[1]
        print(f"Processing file: {filename}")
        
        # Check if the file has a .csv extension
        if not filename.endswith('.csv'):
            print(f"File {filename} is not a CSV file.")
            return JSONResponse({"{filename} is not a CSV file"}, status_code=500)
        
        content = await file.read()
        try:
            # print(bank_type)
            if bank_type == "Costco":
                df = pd.read_csv(io.BytesIO(content), delimiter=',', skiprows=5)
            else:
                df = pd.read_csv(io.BytesIO(content), delimiter=',')
                
            categorized_df = await categorize_transactions(df, bank_type)
            # Ensure the DataFrame is also cleaned before converting to dict
            categorized_df = replace_non_compliant_values(categorized_df)
            processed_dataframes.append(categorized_df.to_dict('records'))
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            return JSONResponse({"error": f"Failed to process {filename}"}, status_code=500)
    return JSONResponse({"data": processed_dataframes})

# Function that uses heuristics to choose input columns and to delete unwanted columns
def column_heuristic(df, bank_type):
    description_col = None
    amount_col = None
    
    print(df.columns)
    df.rename(columns={col: col.strip(string.punctuation + string.whitespace) for col in df.columns}, inplace=True)

    if bank_type == "Chase":
        try:
            del df["Post Date"], df["Category"], df["Type"], df["Memo"]
        except Exception as e:
            print(f"Error finding the column {e} in the DataFrame")
            return JSONResponse({"error": f"The files you uploaded are not accepted"}, status_code=500)
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
        except Exception as e:
            print(f"Error finding the column {e} in the DataFrame")
            return JSONResponse({"error": f"The files you uploaded are not accepted"}, status_code=500)
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
        except Exception as e:
            print(f"Error finding the column {e} in the DataFrame")
            return JSONResponse({"error": f"The files you uploaded are not accepted"}, status_code=500)
    if bank_type == "Penfed":
        try:
            del df["Card Number Last 4"], df["Posted Date"], df["Transaction Type"]
        except Exception as e:
            print(f"Error finding the column {e} in the DataFrame")
            return JSONResponse({"error": f"The files you uploaded are not accepted"}, status_code=500)
    if bank_type == "Visa" or bank_type == "Banco Popular":
        try:
            columns_to_check = ['Date', 'Description', 'Amount']
            are_columns_present = all(column in df.columns for column in columns_to_check)
            if not are_columns_present:
                raise ValueError("The columns are not present in the DataFrame")
        except Exception as e:
                print(f"{e}")
                return JSONResponse({"error": f"The files you uploaded are not accepted"}, status_code=500) 
            
    for col in df.columns:
        if (col.find("Description") != -1) or ('description' in col.lower()):
            description_col = col
        elif (col.find("Amount") != -1) or ('amount' in col.lower()) :
            amount_col = col
        
    return description_col, amount_col

def replace_non_compliant_values(df):
    # Replace 'inf', '-inf' and 'nan' with a compliant value
    df.replace([np.inf, -np.inf, np.nan], [None, None, None], inplace=True)
    return df


app = Starlette(debug=True, routes=[
    Route("/process-csv", endpoint=process_csv_files, methods=["POST"]),
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)   
