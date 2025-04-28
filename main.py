# import pandas as pd
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import optuna
#
# import pandas as pd
# from fuzzywuzzy import process
from db import get_connection
import pandas as pd
import torch
from transformers import AutoTokenizer, T5EncoderModel
# Use a pipeline as a high-level helper
# from transformers import pipeline
from weighted_attention_map import get_sentence_embedding , get_sentence_embedding_master
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import re

# Load the T5 model and tokenizer
# model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = T5EncoderModel.from_pretrained("google-t5/t5-small")

units = {
    "solid": ["gm", "g", "kg", "gram"],
    "liquid": ["ml", "l", "litre"],
    "unit": ["no", "number", "unit"]

}

# Function to get T5 embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy().tolist()

def get_key_by_value(d, value):
    for key, values in d.items():
        if value.lower() in values:
            return key
    return None

# Insert one row into DB
def insert_row(conn, row):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO items (
                itemcode,
                company, company_embedding,
                brand, brand_embedding,
                packaging, packaging_embedding,
                pack_size, pack_size_embedding,
                itemdesc, itemdesc_embedding,filtered_itemdesc_embedding,qty,uom,unit
            ) VALUES (%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s,%s,%s)
        """, (
            row["itemcode"],
            row['company'], row['company_embedding'],
            row['brand'], row['brand_embedding'],
            row['packaging'], row['packaging_embedding'],
            row['pack_size'], row['pack_size_embedding'],
            row['itemdesc'], row['itemdesc_embedding'],row["filtered_itemdesc_embedding"],
            row['qty'],row['uom'],row['unit'],
        ))
    conn.commit()

def insert_embeddings_from_df(df):
    conn = get_connection()
    for idx, row in df.iterrows():
        print(f"Inserting row {idx + 1}/{len(df)}...")
        combine = row['company'] + " " + row['brand'] + " " + row['packaging']+ " " + str(row['qty'])+ " " + row['uomdesc']
        filtered_itemdesc_embedding = get_sentence_embedding(row['itemdesc'],combine)
        itemdesc_embedding = get_sentence_embedding_master(row["itemdesc"])
        # Generate embeddings for all relevant fields
        row_data = {
            'company': row['company'],
            'company_embedding': get_embedding(row['company']),
            'brand': row['brand'],
            'brand_embedding': get_embedding(row['brand']),
            'packaging': row['packaging'],
            'packaging_embedding': get_embedding(row['packaging']),
            'pack_size': row['pack_size'],
            'pack_size_embedding': get_embedding(row['pack_size']),
            'itemdesc':  row['itemdesc'],
            'itemcode': row['itemcode'],
            'qty': int(row['qty']),
            'uom': row['uomdesc'],
            'unit':get_key_by_value(units,row['uomdesc'].lower()),
            'filtered_itemdesc_embedding': filtered_itemdesc_embedding.tolist(), # get_embedding(row['itemdesc']),
            'itemdesc_embedding' : itemdesc_embedding.tolist(),
        }
        # print(row_data)
        # Insert into DB
        insert_row(conn, row_data)

    conn.close()
    print("✅ All rows inserted.")

if __name__ == "__main__":
    master_file = r"D:\Rohit\ORG_SKUR\new_data\master.csv"
    master_df = pd.read_csv(master_file, encoding='ISO-8859-1')
    # Ensure columns are treated as strings
    for col in ["company", "brand", "packaging", "pack_size", "itemdesc","uomdesc"]:
        master_df[col] = master_df[col].astype(str)
    insert_embeddings_from_df(master_df[25000:])




# # Generate embeddings for master_df columns
# print("Generating embeddings for master_df columns...")
# master_df["company_embedding"] = master_df["company"].apply(get_embedding)
# master_df["brand_embedding"] = master_df["brand"].apply(get_embedding)
# master_df["packaging_embedding"] = master_df["packaging"].apply(get_embedding)
# master_df["pack_size_embedding"] = master_df["pack_size"].apply(get_embedding)
# master_df["itemdesc_embedding"] = master_df["itemdesc"].apply(get_embedding)
