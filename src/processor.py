import polars as pl
import re
from config import ALLOWED_PREFIXES
from pathlib import Path
from datasets import load_dataset, VerificationMode
import logging

logger = logging.getLogger(__name__)

# ------ Downloading from HF and prefiltering stage 1 data ------

def prefilter_data(data:pl.LazyFrame) -> pl.LazyFrame:
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
    def _remove_non_alphanumeric(text:str) -> str:
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    data_lf = (
        data
        .filter(pl.col("prefix").str.to_lowercase().str.strip_chars().is_in([p.lower() for p in ALLOWED_PREFIXES])) # filter by allowed prefixes (lowercased)
        .filter(pl.col("src") != pl.col("tgt")) #remove self-translation
        .filter(pl.col("src").str.len_chars() > 0,
                pl.col("tgt").str.len_chars() > 0) #remove empty source or target
        .filter(pl.col("src").str.len_chars() < 1500,
                pl.col("tgt").str.len_chars() < 1500) #remove too long source or target
        .filter(pl.col("src").map_elements(_remove_non_alphanumeric).str.len_chars() > 0,
                pl.col("tgt").map_elements(_remove_non_alphanumeric).str.len_chars() > 0) #remove too many non-alphanumeric characters
        .filter(~pl.col("src").str.starts_with("```"),
                ~pl.col("tgt").str.starts_with("```")) #remove fully code block data
        .filter(pl.col("src").map_elements(_remove_non_alphanumeric).str.len_chars() > 0,
                pl.col("tgt").map_elements(_remove_non_alphanumeric).str.len_chars() > 0) #remove too,cases are like mandarin,tamil,etc.Not needed for dialect MT
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
        verification_mode=VerificationMode.NO_CHECKS
    )
    
    data_lf = pl.LazyFrame(list(dataset['train']))
    cleaned_lf = prefilter_data(data_lf)
    cleaned_df = cleaned_lf.collect()
    
    output_file = output_path / f"{stage.replace('/', '_')}_shard_{shard_idx:05d}.parquet"
    cleaned_df.write_parquet(str(output_file))
    
    return len(cleaned_df)

def clean_text_data(data:pl.LazyFrame) -> pl.LazyFrame:
    """
    Clean text data by:
    - Removing code blocks
    - Removing too many non-alphanumeric characters
    """
    
    def _remove_extra_spaces(text:str) -> str:
        return re.sub(r'\s+', ' ', text.strip())
    
    def _remove_code_blocks(text:str) -> str:
        """
        Remove code blocks from the text.
        It can be in the middle of the text, so we need to remove it.
        """
        pattern = r"```[a-zA-Z0-9]*[\s\S]*?```"
        cleaned = re.sub(pattern, "", text)
        return cleaned.strip()
    
    return data.with_columns(
        src=pl.col("src").map_elements(_remove_code_blocks).map_elements(_remove_extra_spaces),
        tgt=pl.col("tgt").map_elements(_remove_code_blocks).map_elements(_remove_extra_spaces)
    )

# ------ Combining and cleaning text data ------

## Train test splits 

def group_data(data:pl.LazyFrame) -> pl.LazyFrame:
    """
    Group data by semantic key
    """
    result = (
        data
        .unique()
        .with_columns(
            semantic_key=pl.struct(["src", "tgt"]).map_elements(lambda x: "|||".join(sorted([x["src"].strip().lower(), x["tgt"].strip().lower()]))).hash()
        )
        .group_by(
            "semantic_key"
        )
        .agg(
            pl.col("src").count().alias("count")
        )
    )
    return result

def run_data_processing(stage, num_shards, output_dir):
    """Run data processing for a given stage"""
    shard_paths = []
    for shard_idx in range(num_shards):
        count = process_shard(stage, shard_idx, num_shards, output_dir)
        logger.info(f"Processed shard {shard_idx} of {num_shards} for {stage}")
        logger.info(f"Processed {count} examples")
        out_path = f"{output_dir}/{stage.replace('/', '_')}_shard_{shard_idx:05d}.parquet"
        logger.info(f"Saved to {out_path}")
        shard_paths.append(out_path)

    shards:list[pl.LazyFrame] = [pl.scan_parquet(path) for path in shard_paths]    
    combined_lf:pl.LazyFrame = pl.concat(shards, how="vertical_relaxed", rechunk=True)

    cleaned_combined_lf = clean_text_data(combined_lf)
    
    combined_out_path = f"{output_dir}/combined_shards.parquet"
    cleaned_combined_lf.unique().collect().write_parquet(combined_out_path)
    
    # del cleaned_combined_lf #free memory
    
    logger.info(f"Combined {len(shards)} shards into {combined_out_path}")    

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    shards_to_process = {
        # "stage2-part1": 2, #lel this is already filtered by mesolitica
        "stage1":32
    }

    output_dir = (Path(__file__).parent / "data/processed/cleaned_shards").resolve()
    for stage, num_shards in shards_to_process.items():
        run_data_processing(stage, num_shards, output_dir)