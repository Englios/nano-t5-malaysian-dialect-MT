import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
import yaml
from dotenv import load_dotenv
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import set_seed

from dataloader import load_data, build_tokenized_datasets
from metrics import compute_metrics
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for models"})


@dataclass
class DataTrainingArguments:
    train_data_path: str = field(default=f"{Path(__file__).parent}/data/processed/dataset/train.parquet", metadata={"help": "Path to train data file"})
    val_data_path: str = field(default=f"{Path(__file__).parent}/data/processed/dataset/val.parquet", metadata={"help": "Path to val data file"})
    test_data_path: str = field(default=f"{Path(__file__).parent}/data/processed/dataset/test.parquet", metadata={"help": "Path to test data file"})
    max_source_length: int = field(default=1024, metadata={"help": "Max input sequence length"})
    max_target_length: int = field(default=128, metadata={"help": "Max target sequence length"})
    val_max_target_length: Optional[int] = field(default=None, metadata={"help": "Max target length for validation"})
    pad_to_max_length: bool = field(default=False, metadata={"help": "Pad to max length"})
    ignore_pad_token_for_loss: bool = field(default=True, metadata={"help": "Ignore padding in loss"})
    
    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    
    config_dict = None
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            with open(config_path, "r") as f:
                config_content = f.read()
                
                # Handle bash-style ${VAR:-default} syntax
                import re
                def replace_env_vars(match):
                    var_expr = match.group(1)
                    if ":-" in var_expr:
                        var_name, default = var_expr.split(":-", 1)
                        return os.getenv(var_name.strip(), default.strip())
                    else:
                        return os.getenv(var_expr.strip(), match.group(0))
                
                config_content = re.sub(r'\$\{([^}]+)\}', replace_env_vars, config_content)
                config_dict = yaml.safe_load(config_content)
         
            # Workaround: temporarily remove load_best_model_at_end to avoid validation error during parsing
            load_best_wanted = config_dict.pop("load_best_model_at_end", False)
            eval_strategy_wanted = config_dict.pop("evaluation_strategy", "steps")
            
            # Remove wandb-specific keys that aren't part of TrainingArguments
            wandb_config = {
                "wandb_project": config_dict.pop("wandb_project", None),
                "wandb_entity": config_dict.pop("wandb_entity", None),
                "wandb_api_key": config_dict.pop("wandb_api_key", None),
            }
            
            model_args, data_args, training_args = parser.parse_dict(config_dict)
            
            # Now fix the evaluation_strategy if needed and re-enable load_best_model_at_end
            if load_best_wanted and training_args.do_eval:
                from transformers import IntervalStrategy
                # Set evaluation_strategy to match save_strategy
                if eval_strategy_wanted:
                    training_args.evaluation_strategy = IntervalStrategy(eval_strategy_wanted)
                else:
                    training_args.evaluation_strategy = training_args.save_strategy
                training_args.load_best_model_at_end = True
        elif config_path.endswith(".json"):
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(config_path))
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training parameters: {training_args}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"   GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA Version: {torch.version.cuda}")
        logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU :'( ")

    # Initialize wandb (required for logging)
    if "wandb" in (training_args.report_to or []):
        try:
            import wandb
            # Check if wandb is logged in
            if not wandb.api.api_key:
                logger.error("=" * 80)
                logger.error("Weights & Biases is not configured!")
                logger.error("Please run: wandb login")
                logger.error("Or set WANDB_API_KEY in your .env file")
                logger.error("=" * 80)
                raise RuntimeError("Wandb login required. Run 'wandb login' to authenticate.")
            
            # Get wandb config from environment or saved config
            wandb_project = os.getenv("WANDB_PROJECT", "ml-eng-assessment")
            wandb_entity = os.getenv("WANDB_ENTITY")
            wandb_run_name = training_args.run_name if hasattr(training_args, 'run_name') else None
            
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config={
                    "model": model_args.model_name_or_path,
                    "max_source_length": data_args.max_source_length,
                    "max_target_length": data_args.max_target_length,
                    "learning_rate": training_args.learning_rate,
                    "batch_size": training_args.per_device_train_batch_size,
                    "num_epochs": training_args.num_train_epochs,
                }
            )
            logger.info(f"✓ Weights & Biases logging enabled")
            logger.info(f"  Project: {wandb_project}")
            logger.info(f"  Run: {wandb_run_name}")
            if wandb_entity:
                logger.info(f"  Entity: {wandb_entity}")
        except Exception as e:
            logger.error(f"Failed to initialize Weights & Biases: {e}")
            logger.error("Wandb logging is required. Please fix the issue and try again.")
            raise

    set_seed(training_args.seed)

    # ------------------------------ Load data ------------------------------
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    train_data = load_data(data_args.train_data_path)
    val_data = load_data(data_args.val_data_path)
    # test_data = load_data(data_args.test_data_path)

    train_dataset = build_tokenized_datasets(
        train_data,
        tokenizer,
        max_length=data_args.max_source_length,
        data_limit=10000,
        num_proc=training_args.dataloader_num_workers
    )

    eval_dataset = build_tokenized_datasets(
        val_data,
        tokenizer,
        max_length=data_args.max_source_length,
        data_limit=2000,
        num_proc=training_args.dataloader_num_workers
    )

    # ------------------------------ Load model ------------------------------
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Resize embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    # Log model device placement
    if torch.cuda.is_available():
        logger.info(f"Model will be trained on GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("Model will be trained on CPU (no GPU available)")

    # Check decoder_start_token_id
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # ------------------------------ Data collator ------------------------------
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    
    def compute_metrics_wrapper(eval_preds):
        return compute_metrics(eval_preds, tokenizer)

    # ------------------------------ Initialize trainer ------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper if training_args.predict_with_generate else None,
    )

    # ------------------------------ Training ------------------------------
    if training_args.do_train:
        logger.info(f"Starting training. Checkpoints will be saved to: {training_args.output_dir}")
        logger.info(f"  - Save every {training_args.save_steps} steps")
        logger.info(f"  - Keep last {training_args.save_total_limit} checkpoints")
        logger.info(f"  - Best model will be saved based on: {training_args.metric_for_best_model}")
        
        train_result = trainer.train()
        
        # Final model save (this is the best model if load_best_model_at_end is True)
        trainer.save_model()
        logger.info(f"Final model saved to: {training_args.output_dir}")
        
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # ------------------------------ Evaluation ------------------------------
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        max_length = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else data_args.val_max_target_length
        )
        metrics = trainer.evaluate(max_length=max_length, metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ------------------------------ Prediction ------------------------------
    if training_args.do_predict:
        logger.info("*** Predict ***")
        max_length = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else data_args.val_max_target_length
        )
        predict_results = trainer.predict(eval_dataset, metric_key_prefix="predict", max_length=max_length)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)

        if trainer.is_world_process_zero() and training_args.predict_with_generate:
            import numpy as np
            predictions = predict_results.predictions
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                writer.write("\n".join(predictions))


if __name__ == "__main__":
    main()
