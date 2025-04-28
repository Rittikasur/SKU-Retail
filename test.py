# import pandas as pd
# import re
# from sklearn.metrics.pairwise import cosine_similarity
# from weighted_attention_map import get_sentence_embedding_master,get_sentence_embedding
# import numpy as np
# import ast
#
# units = {
#     "solid": ["gm", "g", "kg", "gram"],
#     "liquid": ["ml", "l", "litre"],
#     "unit": ["no", "number", "unit"]
#
# }
#
# def get_key_by_value(d, value):
#     for key, values in d.items():
#         if value.lower() in values:
#             return key
#     return None
#
# def clean_text(text):
#     return re.sub(r"[^a-zA-Z0-9 ]", " ", text)  # Keep only alphanumeric and spaces
#
# data_items_df = pd.read_csv(r"D:\Rohit\ORG_SKUR\new_data\data-new-items-202410.csv")
# master_df = pd.read_csv(r"D:\Rohit\ORG_SKUR\temporary\master_filter_4_dynamic.csv", encoding='ISO-8859-1')
#
# row = data_items_df.iloc[3]
# source_itemdesc = clean_text(row["ITEMDESC"])
#
# combine = row['MANUFACTURE'] + " " + row['BRAND'] + " " + row['PACKTYPE'] + " " + row['PACKSIZE']
# filtered_itemdesc_embedding = get_sentence_embedding(source_itemdesc, combine)
# print(type(filtered_itemdesc_embedding))
#
# itemdesc_similarities = master_df["filtered_itemdesc_embedding"].apply(lambda emb: cosine_similarity([filtered_itemdesc_embedding], [np.array(ast.literal_eval(emb), dtype=float)])[0][0])
#
# if itemdesc_similarities.max() < 0.80:
#     itemdesc_threshold = itemdesc_similarities.max()
# else:
#     itemdesc_threshold = 0.80
#
# master_df['itemdesc_similarity'] = itemdesc_similarities
# master_df_sorted = master_df.sort_values(by='itemdesc_similarity', ascending=False)
#
# print(master_df_sorted.iloc[0]['itemdesc'])


import os
from glob import glob
csv_files = glob(os.path.join("new_data", "*.csv"))
print(csv_files)