import polars as pl
import re
from config import ALLOWED_PREFIXES, DIALECT_PREFIXES
from pathlib import Path
from datasets import load_dataset, VerificationMode
import logging

logger = logging.getLogger(__name__)

# ------ Downloading from HF and prefiltering stage 1 data ------


def prefilter_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Preclean data by:
    - Filtering by allowed prefixes
    - Removing self-translation
    - Removing empty source or target
    - Removing too long source or target
    - Removing too many non-alphanumeric characters,
    - Removing fully code block data that is mostly not translated for the task
    - Removing cases are like mandarin,tamil,etc.Not needed for dialect MT
    """

    def _remove_non_alphanumeric(text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)

    def _remove_extra_spaces(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _remove_code_blocks(text: str) -> str:
        """
        Remove code blocks from the text.
        It can be in the middle of the text, so we need to remove it.
        """
        pattern = r"```[a-zA-Z0-9]*[\s\S]*?```"
        cleaned = re.sub(pattern, "", text)
        return cleaned.strip()

    def _clean_text(text: str) -> str:
        text = _remove_code_blocks(text)
        text = _remove_extra_spaces(text)
        return text

    data_lf = (
        data.with_columns(
            src=pl.col("src").map_elements(_clean_text),
            tgt=pl.col("tgt").map_elements(_clean_text),
        )
        .filter(
            pl.col("prefix")
            .str.to_lowercase()
            .str.strip_chars()
            .is_in([p.lower() for p in ALLOWED_PREFIXES])
        )  # filter by allowed prefixes (lowercased)
        .filter(pl.col("src") != pl.col("tgt"))  # remove self-translation
        .filter(
            pl.col("src").str.len_chars() > 0, pl.col("tgt").str.len_chars() > 0
        )  # remove empty source or target
        .filter(
            pl.col("src").str.len_chars() < 1000, pl.col("tgt").str.len_chars() < 1000
        )  # remove too long source or target
        .filter(
            pl.col("src").map_elements(_remove_non_alphanumeric).str.len_chars() > 10,
            pl.col("tgt").map_elements(_remove_non_alphanumeric).str.len_chars() > 10,
        )  # remove too many non-alphanumeric characters
        .filter(
            ~pl.col("src").str.starts_with("```"), ~pl.col("tgt").str.starts_with("```")
        )  # remove fully code block data
    )

    return data_lf


def process_shard(stage, shard_idx, num_shards, output_dir):
    """Process a single shard, clean it, and save as parquet"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_file = f"{stage}/train-{str(shard_idx).rjust(5, '0')}-of-{str(num_shards).rjust(5, '0')}.parquet"

    dataset = load_dataset(
        "mesolitica/Malaysian-Translation",
        data_files=data_file,
        verification_mode=VerificationMode.NO_CHECKS,
    )

    data_lf = pl.LazyFrame(list(dataset["train"]))
    cleaned_lf = prefilter_data(data_lf)
    cleaned_df = cleaned_lf.collect()

    output_file = (
        output_path / f"{stage.replace('/', '_')}_shard_{shard_idx:05d}.parquet"
    )
    cleaned_df.write_parquet(str(output_file))

    return len(cleaned_df)


# def simple_dialect_detection(data: pl.DataFrame, sample_size: int = 10000) -> pl.DataFrame:
    # """
    # Simple dialect detection by checking if the prefix indicates a dialect.
    # Returns a sample of 100 rows where the prefix matches any dialect prefix.
    # """
    # # Perform filter lazily, then limit to 100 in lazy pipeline.
    # filtered = (
    #     data.filter(
    #         pl.col("prefix")
    #         .str.to_lowercase()
    #         .str.strip_chars()
    #         .is_in([p.lower() for p in DIALECT_PREFIXES])
    #     )
    #     .limit(sample_size)
    #     .select("prefix", "tgt")
    # )
    
    # # for row in filtered.iter_rows(named=True):
    pass
## Train test splits
def create_semantic_key(data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Create semantic key for the data
    """
    return data.with_columns(
        semantic_key=pl.struct(["src"]).map_elements(
            lambda x: x["src"].strip().lower(),
            return_dtype=pl.Utf8
        ).hash()
    )

def create_train_test_split(data: pl.LazyFrame, train_frac: float = 0.8, val_frac: float = 0.1):
    """
    Create train, validation, and test splits by unique semantic_key. 
    Returns: train, val, test DataFrames
    """    
    unique_keys = data.select("semantic_key").unique().collect()
    shuffled_keys:pl.LazyFrame = unique_keys.select(pl.col("semantic_key").shuffle())
    
    n = len(shuffled_keys)
    
    train_idx = int(n * train_frac)
    val_idx = int(n * (train_frac + val_frac))

    train_keys:pl.LazyFrame = shuffled_keys[:train_idx].lazy()
    val_keys:pl.LazyFrame = shuffled_keys[train_idx:val_idx].lazy()
    test_keys:pl.LazyFrame = shuffled_keys[val_idx:].lazy()
    
    train:pl.LazyFrame = data.join(train_keys, on="semantic_key", how="inner").drop("semantic_key")
    val:pl.LazyFrame = data.join(val_keys, on="semantic_key", how="inner").drop("semantic_key")
    test:pl.LazyFrame = data.join(test_keys, on="semantic_key", how="inner").drop("semantic_key")
    
    return train, val, test

def run_data_processing(stage, num_shards, output_dir):
    """Run data processing for a given stage"""
    shard_paths = []
    
    # Process once 
    # for shard_idx in range(num_shards):
    #     # data is already filtered here, so we don't need to prefilter again
    #     count = process_shard(stage, shard_idx, num_shards, output_dir)
    #     logger.info(f"Processed shard {shard_idx} of {num_shards} for {stage}")
    #     logger.info(f"Processed {count} examples")
    #     out_path = (
    #         f"{output_dir}/{stage.replace('/', '_')}_shard_{shard_idx:05d}.parquet"
    #     )
    #     logger.info(f"Saved to {out_path}")
    #     shard_paths.append(out_path)
            
    if len(shard_paths) > 0:
        shards: list[pl.LazyFrame] = [pl.scan_parquet(path) for path in shard_paths]
        combined_lf: pl.LazyFrame = pl.concat(shards, how="vertical_relaxed", rechunk=True)
    else:
        combined_lf = pl.scan_parquet(f"{output_dir}/combined_shards.parquet")
    
    combined_lf = create_semantic_key(combined_lf)
    
    train, val, test = create_train_test_split(combined_lf)
    
    train.sink_parquet(f"{output_dir.parent}/dataset/train.parquet", compression="gzip")
    val.sink_parquet(f"{output_dir.parent}/dataset/val.parquet", compression="gzip")
    test.sink_parquet(f"{output_dir.parent}/dataset/test.parquet", compression="gzip")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    shards_to_process = {
        # "stage2-part1": 2, #lel this is already filtered by mesolitica
        "stage1": 32
    }

    output_dir = (Path(__file__).parent / "data/processed/cleaned_shards").resolve()
    for stage, num_shards in shards_to_process.items():
        run_data_processing(stage, num_shards, output_dir)
