import polars as pl
from transformers import AutoTokenizer
from datasets import Dataset
from config import MODEL_NAME

model_name = MODEL_NAME

def load_data(data_path:str) -> pl.LazyFrame:
    return pl.scan_parquet(data_path)

def tokenize_data(data:pl.LazyFrame) -> pl.LazyFrame:
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    return data.map_elements(lambda x: tokenizer.encode(x))