from starlette.responses import JSONResponse
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.routing import Route
import numpy as np
import pandas as pd
import aiofiles  # For async file operations
import io  # For converting bytes to a format pandas can read
import requests
import uvicorn
import httpx

# Determine an optimal batch size
BATCH_SIZE = 30  # Adjust based on experimentation

# Your existing setup for API interaction
API_URL = "https://thwcvf276mq5w9os.us-east-1.aws.endpoints.huggingface.cloud"

# URL where categories JSON object is stored
json_url = 'https://huggingface.co/Finovatek/Categorization-Model/raw/main/label_to_category_mapping.json'

headers = {
    "Accept": "application/json",
    "Authorization": "Bearer {Add Huggingface Authentication Token here}",
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



async def categorize_transactions(df):
    try:
        # Loading category mappings
        response = requests.get(json_url, headers=headers)
        categories = response.json()

        results = []
        batch_inputs = []
        description_col, amount_col = column_heuristic(df)

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
    print(files)
    processed_dataframes = []
    for file in files:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), delimiter=',')
        try:
            categorized_df = await categorize_transactions(df)
            # Ensure the DataFrame is also cleaned before converting to dict
            categorized_df = replace_non_compliant_values(categorized_df)
            processed_dataframes.append(categorized_df.to_dict('records'))
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            return JSONResponse({"error": f"Failed to process {file.filename}"}, status_code=500)

    return JSONResponse({"data": processed_dataframes})

# Function that uses a simple heuristic to choose input columns or to delete unwanted columns
def column_heuristic(df):
    description_col = None
    amount_col = None
    
    if '<Withdrawal Amount>' in df.columns:
        # Replace NaN values with 0 for calculation
        df['<Withdrawal Amount>'] = df['<Withdrawal Amount>'].fillna(0)
        df['<Deposit Amount>'] = df['<Deposit Amount>'].fillna(0)

        # Ensure correct signs: withdrawals should be negative, deposits should be positive
        df['<Withdrawal Amount>'] = df['<Withdrawal Amount>'].apply(lambda x: -abs(x) if x != 0 else 0)
        df['<Deposit Amount>'] = df['<Deposit Amount>'].apply(lambda x: abs(x) if x != 0 else 0)

        # Create the 'Amount' column by summing the two columns
        df['Amount'] = df['<Withdrawal Amount>'] + df['<Deposit Amount>']
        del df['<Withdrawal Amount>'], df['<Deposit Amount>']
    if "<Additional Info>" in df.columns:
        # Replace the value in Description with the one in Additional Info column if not null
        df.loc[df['<Additional Info>'].notna(), '<Description>'] = df['<Additional Info>']
        del df['<Additional Info>']
        
        
    for col in df.columns:
        if (col.find("Description") != -1) or ('description' in col.lower()):
            description_col = col
        elif (col.find("Amount") != -1) or ('amount' in col.lower()) :
            amount_col = col
        elif (col.find("Category") != -1) or (col.find("Type") != -1) or (col.find("Memo") != -1) or (
            col.find("Post") != -1) or (col.find("Card") != -1) or (col.find("Check") != -1):
            del df[col]
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
