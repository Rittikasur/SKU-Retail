from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch
import re

senmodel = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased")

# Function to clean ITEMDESC (remove special characters)
def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9 ]", " ", text)

def get_sentence_embedding_master(itemdesc):
  embeddings = senmodel.encode(itemdesc, convert_to_tensor=True)
  return embeddings


def get_sentence_embedding(itemdesc,combined):
    itemdesc = clean_text(itemdesc)
    non_overlap_token = soft_attention_for_itemdesc(itemdesc,combined)
    print(f"Unique Tokens: {non_overlap_token} with length {len(non_overlap_token)}")
    if (len(non_overlap_token) < 3):
        print(f"embedding sentence {itemdesc}")
        embeddings = senmodel.encode(itemdesc, convert_to_tensor=True)
    else:
        print(f"embedding sentence {non_overlap_token}")
        embeddings = senmodel.encode(non_overlap_token, convert_to_tensor=True)
    return embeddings

def soft_attention_for_itemdesc(itemdesc,combined_data):
  # Two input sentences
  sentence1 = combined_data # "Transformers are powerful models for NLP."
  sentence2 =  itemdesc # "Transformers help in solving many NLP problems."

  # Tokenize both sentences
  tokens1 = tokenizer.tokenize(sentence1)
  tokens2 = tokenizer.tokenize(sentence2)
  special_tokens = ['[CLS]', '[SEP]']

  # Find common tokens
  common_tokens = (set(tokens1) & set(tokens2)) | set(special_tokens)
  print("Common tokens:", common_tokens)

  # Tokenize second sentence with input IDs
  inputs2 = tokenizer(sentence2, return_tensors="pt")
  input_ids2 = inputs2["input_ids"][0]  # shape: (seq_len,)

  # Map token IDs to tokens
  tokens2_full = tokenizer.convert_ids_to_tokens(input_ids2)
  # Create custom attention mask
  unique_tokens = []
  for tok in tokens2_full:
      if tok not in common_tokens:
          unique_tokens.append(tok)
  token_string = " ".join(unique_tokens)
  return token_string
