
import optuna
import pandas as pd
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, T5EncoderModel
from sklearn.metrics.pairwise import cosine_similarity
from weighted_attention_map import get_sentence_embedding_master,get_sentence_embedding
#from db import get_connection
import ast
import numpy as np

import re
import ollama
import sys
import re


tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = T5EncoderModel.from_pretrained("google-t5/t5-small")
#conn = get_connection()
# Read embeddings and any other relevant columns
#query = "SELECT * FROM items"
#master_df = pd.read_sql(query, conn)
llm_company_list = None
parquet_file = r"D:\Rohit\ORG_SKUR\new_data\processed_embeddings.parquet"
master_df = pd.read_parquet(parquet_file)

units = {
    "solid": ["gm", "g", "kg", "gram"],
    "liquid": ["ml", "l", "litre"],
    "unit": ["no", "number", "unit"]

}


def clean_company_name(name):
    # Lowercase and remove punctuation around business suffixes
    name = name.lower()
    
    # Remove common company suffixes with optional punctuation/brackets
    suffixes = [
        r'\bprivate limited\b',
        r'\blimited\b',
        r'\bltd\b',
        r'\bpvt\b',
        r'\binc\b',
        r'\bcorp\b',
        r'\bco\b',
        r'\bllc\b',
        r'\bplc\b'
    ]
    
    # Remove brackets, dots, and commas around these suffixes
    pattern = re.compile(r'[\(\.\s]*(' + '|'.join(suffixes) + r')[\)\.\s]*', flags=re.IGNORECASE)
    cleaned = pattern.sub('', name)
    
    # Remove remaining standalone punctuation and extra whitespace
    cleaned = re.sub(r'[^\w\s]', '', cleaned)  # remove remaining punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # normalize spaces
    
    return cleaned.upper()



def get_key_by_value(d, value):
    for key, values in d.items():
        if value.lower() in values:
            return key
    return None

# Function to clean ITEMDESC (remove special characters)
def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9 ]", " ", text)  # Keep only alphanumeric and spaces

# Function to get T5 embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy().tolist()

# Function to optimize similarity threshold using Optuna "company"
def optimize_threshold(master_df,embedding_df,col_optimize,source_embedding,trials=20):
    def objective(trial):
        threshold = trial.suggest_float("threshold", 0.5, 1.0)
        similarities = embedding_df.apply(lambda x: cosine_similarity([source_embedding], [np.array(x, dtype=float)])[0][0])
        filtered_df = master_df[similarities >= threshold]
        return abs(filtered_df[col_optimize].nunique() - 1)  # Objective is to get unique company count as close to 1 as possible

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)
    return study.best_params["threshold"]


def filter_similar_products_by_company(input_string: str, df: pd.DataFrame) -> pd.DataFrame:
    matched_companies = set()

    # Work with one representative row per unique company
    for company in df["company"].unique():
        row = df[df["company"] == company].iloc[0]  # pick first row for the company

        prompt = (
            f"Are the following two compnay referring to the same retail product companies?\n\n"
            f"Description 1: {input_string}\n"
            f"Description 2: {row['company']}\n\n"
            f"Answer with a simple 'yes' or 'no'."
        )
        print("-----------------------------------------------------")
        print(prompt)
        print("-----------------------------------------------------")

        response = ollama.chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )

        answer = response["message"]["content"].strip().lower()
        print(answer)

        if "yes" in answer:
            matched_companies.add(company)

    # Filter all rows in the original dataframe with matched companies
    return df[df["company"].isin(matched_companies)].reset_index(drop=True)







def filter_similar_products(input_string: str, df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for index, row in df.iterrows():
        row_master_description = f"Company: {row['company']}, Brand: {row['brand']}, \
        Pack Type: {row['packaging']}, Pack Size: {row['pack_size']}, Item Description : {row['itemdesc']}"

        prompt = (
            f"Are the following two descriptions referring to the same retail product?\n\n"
            f"Description 1: {input_string}\n"
            f"Description 2: {row_master_description}\n\n"
            f"Answer with a simple 'yes' or 'no'."
        )
        print("-----------------------------------------------------")
        print(prompt)
        print("-----------------------------------------------------")

        response = ollama.chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature":0.0}
        )

        answer = response["message"]["content"].strip().lower()
        print(answer)
        if "yes" in answer:
            results.append(True)
        else:
            results.append(False)

    return df[results].reset_index(drop=True)

def process_rows(row):
    # Get values from the first row of data_items_df
    debug = True
    score = 0.0
    reason = ""
    source_manufacture = row["MANUFACTURE"]#.upper()
    source_brand = row["BRAND"]#.upper()
    source_packsize = row["PACKSIZE"]
    source_packtype = row["PACKTYPE"]
    source_itemdesc = clean_text(row["ITEMDESC"])#.upper()
    source_catcode = row["CATEGORY"]#.upper()

    # Check with re that * or x exists, then extract the number and perform multiple or other
    # operations if nedded or else do this operation.
    llm_prompt_qty = int(''.join(re.findall(r'\d+', source_packsize)))
    llm_prompt_uom = ''.join(re.findall(r'[A-Za-z]', source_packsize))

    llm_prompt = f"Company: {source_manufacture}, Brand: {source_brand}, \
    Pack Type: {source_packtype}, Pack Size: {llm_prompt_qty} {llm_prompt_uom}, \
    Item Description : {source_itemdesc}"

    # Generate embeddings for the first row of data_items_df
    print("Generating embedding for data_items_df first row...")
    combine = row['MANUFACTURE'] + " " + row['BRAND'] + " " + row['PACKTYPE'] + " " + row['PACKSIZE']
    filtered_itemdesc_embedding = get_sentence_embedding(source_itemdesc, combine)
    source_itemdesc_emb = get_sentence_embedding_master(source_itemdesc)

    # Step 1: Filter master_df based on company similarity (manufacture)
    print("Filtering master_df based on company (manufacture)...")
    filter_1_run  = True
    while filter_1_run and len(source_manufacture)>1:
        source_manufacture_emb = get_embedding(source_manufacture)
        company_similarities = master_df["company_embedding"].apply(lambda emb: cosine_similarity([source_manufacture_emb], [np.array(emb, dtype=float)])[0][0])
        master_filtered_1 = master_df[company_similarities >= 0.85]  # Adjust threshold if needed
        master_filtered_for_llm =  master_df[company_similarities >= 0.75]  # Adjust threshold if needed
        if master_filtered_1["company"].nunique() < 1:
            source_manufacture = " ".join(source_manufacture[:].split(" ")[:-1])
        else:
            filter_1_run = False

    if debug:
        master_filtered_1.to_csv("temporary/master_filter_1.csv")

    print(f"Computing manuafaturing with company {source_manufacture}")
    dynamic_threshold_company = optimize_threshold(master_filtered_1,master_filtered_1["company_embedding"],"company",source_manufacture_emb,10)
    company_similarities = master_filtered_1["company_embedding"].apply(lambda emb: cosine_similarity([source_manufacture_emb], [np.array(emb, dtype=float)])[0][0])
    master_filtered_1 = master_filtered_1[company_similarities >= dynamic_threshold_company]  # Adjust threshold if needed

    if debug:
        master_filtered_1.to_csv("temporary/master_filter_dynamic_1.csv")

    llm_company_list = master_filtered_for_llm
    if master_filtered_1.shape[0] < 1:
        reason += "Using LLM for company"
        if debug:
            master_filtered_for_llm.to_csv("temporary/input_llm3_for_company.csv")
        matched_company = None
        master_filtered_1 = filter_similar_products_by_company(source_manufacture,master_filtered_for_llm)
        master_filtered_1['clean_company'] = master_filtered_1["company"].apply(lambda row: clean_company_name(row))
        master_filtered_1['clean_company_embedding'] =  master_filtered_1["clean_company"].apply(lambda row: get_embedding(row))
        source_manufacture_emb = get_embedding(clean_company_name(source_manufacture))
        dynamic_threshold_company = optimize_threshold(master_filtered_1,master_filtered_1["clean_company_embedding"],"company",source_manufacture_emb,10)
        company_similarities = master_filtered_1["clean_company_embedding"].apply(lambda emb: cosine_similarity([source_manufacture_emb], [np.array(emb, dtype=float)])[0][0])
        master_filtered_1 = master_filtered_1[company_similarities >= dynamic_threshold_company]  # Adjust threshold if needed
        if debug:
            master_filtered_1.to_csv("temporary/llm_filter_1.csv")
        if master_filtered_1.shape[0] < 1:
            matched_company = None
            reason += " | Target company not found"
        else:
            matched_company = '-'.join(master_filtered_1['company'].dropna().astype(str).unique())
    else:
        matched_company = master_filtered_1['company'].unique()[0]
        score += 0.20

    # sys.exit(0)

    # Step 2: Filter the result based on brand similarity
    print("Filtering master_df based on brand...")
    filter_2_run = True
    while filter_2_run and len(source_brand)>1:
        source_brand_emb = get_embedding(source_brand)
        brand_similarities = master_filtered_1["brand_embedding"].apply(lambda emb: cosine_similarity([source_brand_emb], [np.array(emb, dtype=float)])[0][0])
        master_filtered_2 = master_filtered_1[brand_similarities >= 0.85]
        master_filtered_2_for_llm = master_filtered_1[brand_similarities >= 0.75]

        if master_filtered_2["brand"].nunique() < 1:
            source_brand = " ".join(source_brand[:].split(" ")[:-1])
        else:
            filter_2_run = False

    if debug:
        master_filtered_2.to_csv("temporary/master_filter_2.csv")


    print(f"Computing manuafaturing with brand {source_brand}")
    dynamic_threshold_brand = optimize_threshold(master_filtered_2,master_filtered_2["brand_embedding"],"brand",source_brand_emb,10)
    brand_similarities = master_filtered_2["brand_embedding"].apply(lambda emb: cosine_similarity([source_brand_emb], [np.array(emb, dtype=float)])[0][0])
    master_filtered_2 = master_filtered_2[brand_similarities >= dynamic_threshold_brand]  # Adjust threshold if needed

    if debug:
        master_filtered_2.to_csv("temporary/master_filter_2_dynamic.csv")

    if master_filtered_2.shape[0] < 1:
        reason += "Using LLM for Brand"
        matched_brand = None
        master_filtered_2 = filter_similar_products(llm_prompt, master_filtered_2_for_llm)
        if debug:
            master_filtered_2.to_csv("temporary/llm_filter_2.csv")
        if master_filtered_2.shape[0] < 1:
            matched_brand = None
            reason += " | Target Brand not found"
        else:
            matched_brand = '-'.join(master_filtered_2['brand'].dropna().astype(str).unique())
    else:
        matched_brand = master_filtered_2['brand'].unique()[0]
        score += 0.20

    # Step 3: Filter the result based on packtype similarity
    print("Filtering master_df based on packtype...")
    filter_3_run = True
    while filter_3_run and len(source_packtype)>1:
        source_packtype_emb = get_embedding(source_packtype)
        packtype_similarities = master_filtered_2["packaging_embedding"].apply(lambda emb: cosine_similarity([source_packtype_emb], [np.array(emb, dtype=float)])[0][0])
        master_filtered_3 = master_filtered_2[packtype_similarities >= 0.5]
        if master_filtered_3["packaging"].nunique() < 1:
            source_packtype = " ".join(source_packtype[:].split(" ")[:-1])
        else:
            filter_3_run = False

    if debug:
        master_filtered_3.to_csv("temporary/master_filter_3.csv")

    print(f"Computing manuafaturing with packtype {source_packtype}")
    dynamic_threshold_packtype = optimize_threshold(master_filtered_3,master_filtered_3["packaging_embedding"],"packaging",source_packtype_emb,80)
    packtype_similarities = master_filtered_3["packaging_embedding"].apply(lambda emb: cosine_similarity([source_packtype_emb], [np.array(emb, dtype=float)])[0][0])
    master_filtered_3 = master_filtered_3[packtype_similarities >= dynamic_threshold_packtype]  # Adjust threshold if needed

    if debug:
        master_filtered_3.to_csv("temporary/master_filter_3_dynamic.csv")

    if master_filtered_3.shape[0] < 1:
        reason += "| Target Packtype not found"
        matched_packtype = None
    else:
        matched_packtype = master_filtered_3['packaging'].unique()[0]
        score += 0.20

    # Step 4: Filter the result based on packsize similarity
    print("Filtering master_df based on packsize...")
    packsize = source_packsize
    qty = int(''.join(re.findall(r'\d+', packsize)))
    uom = ''.join(re.findall(r'[A-Za-z]', packsize))
    unit = get_key_by_value(units,uom.lower())
    temp_filter_4_df = master_filtered_3[master_filtered_3["qty"] == qty]
    master_filtered_4 = temp_filter_4_df[temp_filter_4_df['unit'] == unit]

    if debug:
        master_filtered_4.to_csv("temporary/master_filter_4_dynamic.csv")

    if master_filtered_4.shape[0] < 1:
        reason += "| Target Packsize or UOM not found"
        matched_packsize = None
    else:
        matched_packsize = str(qty) + uom
        score += 0.20

    # Step 5: Filter the result based on itemdesc similarity
    print("Filtering master_df based on item description...")

    itemdesc_similarities = master_filtered_4["filtered_itemdesc_embedding"].apply(lambda emb: cosine_similarity([filtered_itemdesc_embedding], [np.array(emb, dtype=float)])[0][0])
    master_filtered_4['itemdesc_similarity'] = itemdesc_similarities
    master_final_filtered = master_filtered_4[itemdesc_similarities >= 0.80]

    if debug:
        master_final_filtered.to_csv("temporary/master_filter_final.csv")

    if master_final_filtered.shape[0] < 1:
        reason += "| Target Itemdesc not found"
        matched_itemdesc = None
    else:
        matched_itemdesc = master_final_filtered['itemdesc'].unique()[0]

    # Print final filtered results
    print("\nFinal Filtered Master Data:")
    print(master_final_filtered)




    if master_final_filtered.shape[0] < 1:
        master_filtered_4_sorted = master_filtered_4.sort_values(by='itemdesc_similarity', ascending=False)
        suggested_match = master_filtered_4_sorted.iloc[0] if not master_filtered_4_sorted.empty else None
        return {
            "PERIOD"	: row['PERIOD'],
            "AUDITTYPE" : row['AUDITTYPE'],	
            "STORECODE":  row['STORECODE'],	
            "DLRCODE": row['DLRCODE'] ,	
            "ITEMCODE":  row['ITEMCODE'],

        "ITEMDESC": row["ITEMDESC"],
        "Brand": row["BRAND"],
        "Company": row["MANUFACTURE"],
        "Packtype": row['PACKTYPE'],
        "Packsize": row['PACKSIZE'],
       

        "Matched_ITEMDESC":  matched_itemdesc,
        "Matched_BRAND":  matched_brand,
        "Matched_MANUFACTURE":  matched_company,
        "Matched_PACKTYPE":  matched_packtype,
        "Matched_PACKSIZE":  matched_packsize,
        "Reason" : reason,
        "Suggestion": suggested_match["itemdesc"] if suggested_match is not None else None,
        "Suggestion_code": suggested_match["itemcode"] if suggested_match is not None else None,
        "Score" : score + float(suggested_match["itemdesc_similarity"])*0.2 if suggested_match is not None else None
    }
    master_final_filtered = master_final_filtered.sort_values(by='itemdesc_similarity', ascending=False)
    best_match = master_final_filtered.iloc[0] if not master_final_filtered.empty else None
    return {
        "PERIOD"	: row['PERIOD'],
        "AUDITTYPE" : row['AUDITTYPE'],	
        "STORECODE":  row['STORECODE'],	
        "DLRCODE": row['DLRCODE'] ,	
        "ITEMCODE":  row['ITEMCODE'],
        "ITEMDESC": row["ITEMDESC"],
        "Brand": row["BRAND"],
        "Company": row["MANUFACTURE"],
        "Packtype": row['PACKTYPE'],
        "Packsize": row['PACKSIZE'],


        "Matched_ITEMDESC": best_match["itemdesc"] if best_match is not None else None,
        "Matched_BRAND": best_match["brand"] if best_match is not None else None,
        "Matched_MANUFACTURE": best_match["company"] if best_match is not None else None,
        "Matched_PACKTYPE":  best_match["packaging"] if best_match is not None else None,
        "Matched_PACKSIZE":  best_match["pack_size"] if best_match is not None else None,
        "Reason" : reason,
        "Suggestion": "EMF",
        "Suggestion_code":  best_match["itemcode"] if best_match is not None else None,
        "Score" : score + float(best_match["itemdesc_similarity"])*0.2 if best_match is not None else None
    }

def process_csv(filename,startindex,endindex):
    # Process all rows in data_items_df
    data_items_df = pd.read_csv(f"D:/Rohit/ORG_SKUR/new_data/data_file/{filename}")
    results = []
    for index, row in data_items_df.iloc[startindex:endindex].iterrows():
        print(f"Processing row {index + 1}/{len(data_items_df)}...")
        try:
            x = process_rows(row)
            results.append(x)
        except Exception as e:
            print("-------------------Error------------")
            results.append({
                "PERIOD"	: row['PERIOD'],
                "AUDITTYPE" : row['AUDITTYPE'],	
                "STORECODE":  row['STORECODE'],	
                "DLRCODE": row['DLRCODE'] ,	
                "ITEMCODE":  row['ITEMCODE'],
                "ITEMDESC": row["ITEMDESC"],
                "Brand": row["BRAND"],
                "Company": row["MANUFACTURE"],
                "Packtype": row['PACKTYPE'],
                "Packsize": row['PACKSIZE'],


                "Matched_ITEMDESC":  None,
                "Matched_BRAND": None,
                "Matched_MANUFACTURE":  None,
                "Matched_PACKTYPE":  None,
                "Matched_PACKSIZE":  None,
                "Reason" : None,
                "Suggestion": "None-Except",
                "Suggestion_code":   None,
                "Score" :  None
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"D:/AILABS/Projects/Clients/SKU-R/application/processed_result_chunk_3/tmp_nov_{filename}", index=False)


if __name__=="__main__":
    files = [#"october.csv",
              "data-new-items-202411.csv",
             # "data-new-items-202412.csv",
             # "data-new-items-202501.csv",
              #"data-new-items-202502.csv",
             # "data-new-items-202503.csv",
             ]
    start = 1
    end = 2
    print("Starting process")
    for file in files:
        process_csv(file,start,end)