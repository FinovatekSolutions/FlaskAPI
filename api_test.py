from starlette.responses import JSONResponse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route
import pandas as pd
import aiofiles  # For async file operations
import io  # For converting bytes to a format pandas can read
import requests
import uvicorn

# Determine an optimal batch size
BATCH_SIZE = 30  # Adjust based on experimentation

# Your existing setup for API interaction
API_URL = "https://thwcvf276mq5w9os.us-east-1.aws.endpoints.huggingface.cloud"

# URL where categories JSON object is stored
json_url = 'https://huggingface.co/Finovatek/Categorization-Model/raw/main/label_to_category_mapping.json'

headers = {
    "Accept": "application/json",
    "Authorization": "Bearer hf_XXXX",
    "Content-Type": "application/json" 
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


async def categorize_transactions(df):
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
        
        # Assuming the response should be a list with one entry per input
        if isinstance(response, list):
            for resp in response:
                # Process each response, adjusting according to the actual structure
                # !!!!TODO: Categories are not going to be mapped so we have to change reverse mapping 
                if 'label' in resp:
                    number = resp['label'].split('_')[1]
                    category = categories.get(number, 'Unknown Category')
                else:
                    category = 'Unknown Category'  # Fallback for unexpected response structure
                results.append(category)
        else:
            # If response is not as expected, log and append placeholder values
            print(f"Unexpected response format or error for batch starting at index {batch_start}")
            for _ in range(len(batch_inputs)):
                results.append('Unknown Category')

    # Before assigning, check the lengths match
    assert len(df) == len(results), f"Mismatch in DataFrame length and results: {len(df)} vs {len(results)}"
    df['Category'] = results
    return df

# Function that receives HTTP request with form data 
async def process_csv_files(request: Request):
    form = await request.form()
    csv_files = [file for _, file in form.items() if file.filename.endswith(".csv")]
    
    processed_dataframes = []
    for file in csv_files:
        async with aiofiles.open(file.file, mode = 'rb') as af:
            content = await af.read()
            df = pd.read_csv(io.BytesIO(content), delimiter=',')
            categorized_df = await categorize_transactions(df)
            processed_dataframes.append(categorized_df.to_dict('records'))

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


app = Starlette(debug=True, routes=[
    Route("/process-csv", endpoint=process_csv_files, methods=["POST"]),
])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)    

            
