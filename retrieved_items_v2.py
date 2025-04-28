
import optuna
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, T5EncoderModel
from sklearn.metrics.pairwise import cosine_similarity
from weighted_attention_map import get_sentence_embedding_master,get_sentence_embedding
from db import get_connection
import ast
import numpy as np
import re
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = T5EncoderModel.from_pretrained("google-t5/t5-small")
conn = get_connection()
# Read embeddings and any other relevant columns
query = "SELECT * FROM items"
master_df = pd.read_sql(query, conn)

units = {
    "solid": ["gm", "g", "kg", "gram"],
    "liquid": ["ml", "l", "litre"],
    "unit": ["no", "number", "unit"]

}

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
        similarities = embedding_df.apply(lambda x: cosine_similarity([source_embedding], [np.array(ast.literal_eval(x), dtype=float)])[0][0])
        filtered_df = master_df[similarities >= threshold]
        return abs(filtered_df[col_optimize].nunique() - 1)  # Objective is to get unique company count as close to 1 as possible

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)
    return study.best_params["threshold"]

# Jaccard Similarity Calculation
def jaccard_similarity(str1, str2):
    words1 = set(str1.split())
    words2 = set(str2.split())

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)

# Jaccard Similarity Calculation
def jaccard_similarity_for_packtype(str1, str2):
    words1 = set(str1)
    words2 = set(str2)
    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)


def process_rows(row):
    # Get values from the first row of data_items_df
    debug = False
    score = 0.0
    reason = ""
    source_manufacture = row["MANUFACTURE"]
    source_brand = row["BRAND"]
    source_packsize = row["PACKSIZE"]
    source_packtype = row["PACKTYPE"]
    source_itemdesc = clean_text(row["ITEMDESC"])

    # Generate embeddings for the first row of data_items_df
    print("Generating embedding for data_items_df first row...")
    combine = row['MANUFACTURE'] + " " + row['BRAND'] + " " + row['PACKTYPE'] + " " + row['PACKSIZE']
    filtered_itemdesc_embedding = get_sentence_embedding(source_itemdesc, combine)
    source_itemdesc_emb = get_sentence_embedding_master(source_itemdesc)

    print()
    # Step 1: Filter master_df based on company similarity (manufacture)
    print("Filtering master_df based on company (manufacture)...")
    filter_1_run  = True
    while filter_1_run and len(source_manufacture)>1:
        source_manufacture_emb = get_embedding(source_manufacture)
        company_similarities = master_df["company_embedding"].apply(lambda emb: cosine_similarity([source_manufacture_emb], [np.array(ast.literal_eval(emb), dtype=float)])[0][0])
        master_filtered_1 = master_df[company_similarities >= 0.85]  # Adjust threshold if needed
        if master_filtered_1["company"].nunique() < 1:
            source_manufacture = " ".join(source_manufacture[:].split(" ")[:-1])
        else:
            filter_1_run = False

    if debug:
        master_filtered_1.to_csv("temporary/master_filter_1.csv")

    print(f"Computing manuafaturing with company {source_manufacture}")
    dynamic_threshold_company = optimize_threshold(master_filtered_1,master_filtered_1["company_embedding"],"company",source_manufacture_emb,10)
    company_similarities = master_filtered_1["company_embedding"].apply(lambda emb: cosine_similarity([source_manufacture_emb], [np.array(ast.literal_eval(emb), dtype=float)])[0][0])
    master_filtered_1 = master_filtered_1[company_similarities >= dynamic_threshold_company]  # Adjust threshold if needed

    if debug:
        master_filtered_1.to_csv("temporary/master_filter_dynamic_1.csv")

    if master_filtered_1.shape[0] < 1:
        reason += "Target Company not found"
        matched_company = None
    else:
        matched_company = master_filtered_1['company'].unique()[0]
        score += 0.20

    # Step 2: Filter the result based on brand similarity
    print("Filtering master_df based on brand...")
    filter_2_run = True
    while filter_2_run and len(source_brand)>1:
        source_brand_emb = get_embedding(source_brand)
        brand_similarities = master_filtered_1["brand_embedding"].apply(lambda emb: cosine_similarity([source_brand_emb], [np.array(ast.literal_eval(emb), dtype=float)])[0][0])
        master_filtered_2 = master_filtered_1[brand_similarities >= 0.85]
        if master_filtered_2["brand"].nunique() < 1:
            source_brand = " ".join(source_brand[:].split(" ")[:-1])
        else:
            filter_2_run = False

    if debug:
        master_filtered_2.to_csv("temporary/master_filter_2.csv")


    print(f"Computing manuafaturing with brand {source_brand}")
    dynamic_threshold_brand = optimize_threshold(master_filtered_2,master_filtered_2["brand_embedding"],"brand",source_brand_emb,80)
    brand_similarities = master_filtered_2["brand_embedding"].apply(lambda emb: cosine_similarity([source_brand_emb], [np.array(ast.literal_eval(emb), dtype=float)])[0][0])
    master_filtered_2 = master_filtered_2[brand_similarities >= dynamic_threshold_brand]  # Adjust threshold if needed

    if debug:
        master_filtered_2.to_csv("temporary/master_filter_2_dynamic.csv")

    if master_filtered_2.shape[0] < 1:
        reason += "| Target Brand not found"
        matched_brand = None
    else:
        matched_brand = master_filtered_2['brand'].unique()[0]
        score += 0.20

    # Step 3: Filter the result based on packtype similarity
    print("Filtering master_df based on packtype...")
    filter_3_run = True
    while filter_3_run and len(source_packtype)>1:
        source_packtype_emb = get_embedding(source_packtype)
        packtype_similarities = master_filtered_2["packaging_embedding"].apply(lambda emb: cosine_similarity([source_packtype_emb], [np.array(ast.literal_eval(emb), dtype=float)])[0][0])
        master_filtered_3 = master_filtered_2[packtype_similarities >= 0.5]
        if master_filtered_3["packaging"].nunique() < 1:
            source_packtype = " ".join(source_packtype[:].split(" ")[:-1])
        else:
            filter_3_run = False

    if debug:
        master_filtered_3.to_csv("temporary/master_filter_3.csv")



    print(f"Computing manuafaturing with packtype {source_packtype}")
    dynamic_threshold_packtype = optimize_threshold(master_filtered_3,master_filtered_3["packaging_embedding"],"packaging",source_packtype_emb,80)
    packtype_similarities = master_filtered_3["packaging_embedding"].apply(lambda emb: cosine_similarity([source_packtype_emb], [np.array(ast.literal_eval(emb), dtype=float)])[0][0])
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

    itemdesc_similarities = master_filtered_4["filtered_itemdesc_embedding"].apply(lambda emb: cosine_similarity([filtered_itemdesc_embedding], [np.array(ast.literal_eval(emb), dtype=float)])[0][0])
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
        "ITEMDESC": row["ITEMDESC"],
        "Brand": row["BRAND"],
        "Company": row["MANUFACTURE"],
        "Packtype": row['PACKTYPE'],
        "Packsize":row['PACKSIZE'],

        "Matched_ITEMDESC":  matched_itemdesc,
        "Matched_BRAND":  matched_brand,
        "Matched_MANUFACTURE":  matched_company,
        "Matched_PACKTYPE":  matched_packtype,
        "Matched_PACKSIZE":  matched_packsize,
        "Reason" : reason,
        "Suggestion": suggested_match["itemdesc"] if suggested_match is not None else None,
        "Score" : score + float(suggested_match["itemdesc_similarity"])*0.2 if suggested_match is not None else None
    }
    master_final_filtered = master_final_filtered.sort_values(by='itemdesc_similarity', ascending=False)
    best_match = master_final_filtered.iloc[0] if not master_final_filtered.empty else None
    return {
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
            print(str(e))

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"D:/Rohit/ORG_SKUR/processed_result_chunk_3/filtered_results_{filename}", index=False)


if __name__=="__main__":
    files = ["data-new-items-202410.csv",
             "data-new-items-202411.csv",
             "data-new-items-202412.csv",
             "data-new-items-202501.csv",
             "data-new-items-202502.csv",
             "data-new-items-202503.csv",
             ]
    start = 30
    end = 40
    print("Starting process")
    for file in files:
        process_csv(file,start,end)