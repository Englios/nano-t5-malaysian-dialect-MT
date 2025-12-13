import polars as pl
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from datasets import Dataset

def load_data(data_path: str) -> pl.LazyFrame:
    return pl.scan_parquet(data_path)

def build_tokenized_datasets(
    data: pl.LazyFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 1024,
    data_limit: int | None = None,
    num_proc: int = 8,
    seed: int = 67,
    shuffle: bool = False,
) -> Dataset:
    """Build tokenized datasets from polars LazyFrame."""
    dataframe = (
        data.with_columns(
            input_text=pl.format("terjermah {} ke {}: ", pl.col("detected_src").str.to_lowercase(), pl.col("dialect").str.to_lowercase()),
            target_text=pl.col("tgt"),
        )
        .filter(
            pl.col("input_text").str.len_chars() <= max_length,
            pl.col("target_text").str.len_chars() <= max_length
        )
    )
    
    if shuffle:
        dataframe = dataframe.select(pl.all()).collect().sample(fraction=1.0, shuffle=True, seed=seed)

    if data_limit is not None:
        dataframe = dataframe.limit(data_limit)

    dataframe = dataframe.select("input_text", "target_text")
    
    dataset = Dataset.from_polars(dataframe)
    
    def _tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=128,
            padding="max_length",
            truncation=True,
        )
        
        labels = tokenizer(
            text_target=examples["target_text"],
            max_length=128,
            padding="max_length",
            truncation=True,
        )
        
        if "input_ids" in labels:
            labels["input_ids"] = [
                [(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in label] 
                for label in labels["input_ids"]
            ]
        
        model_inputs["labels"] = [
            [int(x) for x in label] for label in labels["input_ids"]
        ]
        
        if "token_type_ids" in model_inputs:
            del model_inputs["token_type_ids"]
        
        return model_inputs

    tokenized_dataset = dataset.map(
        _tokenize_function, 
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names 
    )
    
    return tokenized_dataset

if __name__ == "__main__":
    from config import MODEL_NAME
    from pathlib import Path
    from transformers import AutoTokenizer
    
    data_path = Path(__file__).parent / "data" / "processed" / "dataset" / "train.parquet"
    data = load_data(str(data_path))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_dataset = build_tokenized_datasets(data, tokenizer, data_limit=10000, seed=67, shuffle=True)
    print(tokenized_dataset)

