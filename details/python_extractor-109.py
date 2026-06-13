import pdfplumber
import pandas as pd
import json
import re
import os


# Ensure this matches the exact filename you uploaded
PDF_FILENAME = '/kaggle/input/hivdrug/NYSDOH-AI-Drug-Drug-Interaction-Guide_10-7-2025_HG.pdf'

OUTPUT_CSV = 'hiv_drug_interactions_data.csv'
OUTPUT_JSON = 'hiv_drug_interactions_data.json'


def fix_duplicate_columns(df):

    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def clean_table_data(table):

    if not table:
        return None

    if len(table) < 2:
        return None

    df = pd.DataFrame(table[1:], columns=table[0])
    
    df.dropna(how='all', inplace=True)
    
    df = fix_duplicate_columns(df)
    
    df.ffill(inplace=True)
    

    if hasattr(df, 'map'):
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    else:
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    return df

def find_table_title(page_text):


    # Regex pattern to find "Table X: Title" or "Table XA: Title"
    match = re.search(r'(Table\s+\d+[A-Za-z]*:\s*.*)', page_text)
    
    if match:
        return match.group(1)
    
    return "Untitled Table"

def extract_interactions(pdf_path):
    
    all_dataframes = []
    json_data = {}

    print(f"Starting extraction from: {pdf_path}...")

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            current_title = find_table_title(page_text)
            
            # Extract tables from the page
            tables = page.extract_tables()
            
            
            # Process each table found on the page
            for table_idx, table in enumerate(tables):
                cleaned_df = clean_table_data(table)
                
                if cleaned_df is not None and not cleaned_df.empty:
                    cleaned_df['Source_Page'] = i + 1
                    cleaned_df['Table_Title'] = current_title
                    cleaned_df['Table_Index_On_Page'] = table_idx
                    
                    all_dataframes.append(cleaned_df)
                    
                    if current_title not in json_data:
                        json_data[current_title] = []
                    
                    json_data[current_title].append(cleaned_df.to_dict(orient='records'))
                    
                    print(f"Page {i+1}: Extracted table '{current_title}' with {len(cleaned_df)} rows.")

    return all_dataframes, json_data

def save_data(dataframes, json_structure, csv_name, json_name):

    master_df = pd.concat(dataframes, ignore_index=True)
    master_df.to_csv(csv_name, index=False)
    print(f"Saved combined data to: {csv_name}")

    with open(json_name, 'w') as f:
        json.dump(json_structure, f, indent=4)
    print(f"Saved structured JSON to: {json_name}")


dfs, data_json = extract_interactions(PDF_FILENAME)

save_data(dfs, data_json, OUTPUT_CSV, OUTPUT_JSON)

print("\nExtraction Complete.")
print(f"Total tables extracted: {len(dfs)}")
