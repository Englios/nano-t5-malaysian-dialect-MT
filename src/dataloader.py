import polars as pl
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from config import MODEL_NAME
from pathlib import Path

from datasets import Dataset

model_name = MODEL_NAME
tokenizer_model = AutoTokenizer.from_pretrained(model_name)

def load_data(data_path:str) -> pl.LazyFrame:
    return pl.scan_parquet(data_path)

def build_tokenized_datasets(
    data:pl.LazyFrame,
    tokenizer:PreTrainedTokenizerBase,
    max_characters:int = 1024,
    data_limit:int | None = None,
) -> Dataset:
    
    # Preprocess data
    dataframe = (
        data.with_columns(
            input_text=pl.concat_str([pl.col("prefix").str.strip_chars(),pl.col("src").str.strip_chars()], separator=" "),
            target_text=pl.col("tgt"),
        )
        .filter(pl.col("input_text").str.len_chars() <= max_characters, pl.col("target_text").str.len_chars() <= max_characters)
        .select("input_text", "target_text")
        .collect()
    )
    if data_limit is not None:
        dataframe = dataframe.limit(data_limit)
    
    # Convert to Dataset
    dataset = Dataset.from_polars(dataframe)
    
    # Define a tokenize_function for use with map
    def _tokenize_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=max_characters,
            padding="max_length",
            truncation=True,
        )
        
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=examples["target_text"],
            max_length=max_characters,
            padding="max_length",
            truncation=True,
        )
        
        # Replace pad_token_id with -100 in labels to ignore padding in loss
        if "input_ids" in labels:
            labels["input_ids"] = [
                [(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in label] 
                for label in labels["input_ids"]
            ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(_tokenize_function, batched=True)
    
    return tokenized_dataset

if __name__ == "__main__":
    data_path = Path(__file__).parent / "data" / "processed" / "dataset" / "train.parquet"
    print(data_path)
    data = load_data(str(data_path))
    model_inputs = build_tokenized_datasets(data, tokenizer_model, data_limit=10000)
    print(model_inputs)
    print(model_inputs[0])
    