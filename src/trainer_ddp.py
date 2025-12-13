import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import re

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


import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

logger = logging.getLogger(__name__)

def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Cache directory for models"}
    )

@dataclass
class DataTrainingArguments:
    train_data_path: str = field(
        default=f"{Path(__file__).parent}/data/processed/dataset/train.parquet",
        metadata={"help": "Path to train data file"},
    )
    val_data_path: str = field(
        default=f"{Path(__file__).parent}/data/processed/dataset/val.parquet",
        metadata={"help": "Path to val data file"},
    )
    test_data_path: str = field(
        default=f"{Path(__file__).parent}/data/processed/dataset/test.parquet",
        metadata={"help": "Path to test data file"},
    )
    max_source_length: int = field(
        default=1024, metadata={"help": "Max input sequence length"}
    )
    max_target_length: int = field(
        default=128, metadata={"help": "Max target sequence length"}
    )
    val_max_target_length: Optional[int] = field(
        default=None, metadata={"help": "Max target length for validation"}
    )
    pad_to_max_length: bool = field(
        default=False, metadata={"help": "Pad to max length"}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True, metadata={"help": "Ignore padding in loss"}
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def parse_config(config_path=None):
    """
    Parse configuration from YAML/JSON file or command line arguments.
    Returns: model_args, data_args, training_args, train_size, val_size, test_size, wandb_config
    """
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    config_dict = None
    train_size_wanted = None
    val_size_wanted = None
    test_size_wanted = None
    wandb_config = {
        "wandb_project": None,
        "wandb_entity": None,
        "wandb_api_key": None,
    }
    
    if config_path is None:
        config_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if config_path:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            with open(config_path, "r") as f:
                config_content = f.read()
                
                def replace_env_vars(match):
                    var_expr = match.group(1)
                    if ":-" in var_expr:
                        var_name, default = var_expr.split(":-", 1)
                        return os.getenv(var_name.strip(), default.strip())
                    else:
                        return os.getenv(var_expr.strip(), match.group(0))

                config_content = re.sub(
                    r"\$\{([^}]+)\}", replace_env_vars, config_content
                )
                config_dict = yaml.safe_load(config_content)

                load_best_wanted = config_dict.pop("load_best_model_at_end", False)
                eval_strategy_wanted = config_dict.pop("evaluation_strategy", "steps")
                train_size_wanted = config_dict.pop("train_size", None)
                val_size_wanted = config_dict.pop("val_size", None)
                test_size_wanted = config_dict.pop("test_size", None)

                wandb_config = {
                    "wandb_project": config_dict.pop("wandb_project", None),
                    "wandb_entity": config_dict.pop("wandb_entity", None),
                    "wandb_api_key": config_dict.pop("wandb_api_key", None),
                }

                model_args, data_args, training_args = parser.parse_dict(config_dict)

                if load_best_wanted and training_args.do_eval:
                    from transformers import EvalStrategy
                    
                    if eval_strategy_wanted:
                        training_args.eval_strategy = EvalStrategy(eval_strategy_wanted)
                    else:
                        training_args.eval_strategy = training_args.save_strategy
                    training_args.load_best_model_at_end = True
        elif config_path.endswith(".json"):
            model_args, data_args, training_args = parser.parse_json_file(
                json_file=os.path.abspath(config_path)
            )
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    return model_args, data_args, training_args, train_size_wanted, val_size_wanted, test_size_wanted, wandb_config


def main_worker(rank, world_size, config_path=None):
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    load_dotenv()
    
    model_args, data_args, training_args, train_size_wanted, val_size_wanted, test_size_wanted, wandb_config = parse_config(config_path)
    
    if rank == 0:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.WARNING,
        )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if rank == 0:
        logger.info(f"Training parameters: {training_args}")
        num_gpus = torch.cuda.device_count()
        logger.info(f"   Total GPUs available: {num_gpus}")
        for i in range(num_gpus):
            logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"   GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        logger.info(f"   CUDA Version: {torch.version.cuda}")
        if world_size > 1:
            logger.info(f"   Distributed training (DDP) is ACTIVE")
            logger.info(f"   World size: {world_size}")
            logger.info(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size}")

    if "wandb" in (training_args.report_to or []):
        if rank == 0:
            import wandb
            if not wandb.api.api_key:
                logger.error("=" * 80)
                logger.error("Weights & Biases is not configured!")
                logger.error("Please run: wandb login")
                logger.error("Or set WANDB_API_KEY in your .env file")
                logger.error("=" * 80)
                raise RuntimeError("Wandb login required. Run 'wandb login' to authenticate.")

            if wandb.run is None:
                wandb_project = wandb_config.get("wandb_project") or os.getenv("WANDB_PROJECT", "ml-eng-assessment")
                wandb_entity = wandb_config.get("wandb_entity") or os.getenv("WANDB_ENTITY")
                wandb_run_name = training_args.run_name if hasattr(training_args, "run_name") else None

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
                    },
                )
            else:
                logger.info("  Wandb already initialized by sweep agent")

            if wandb.config:
                sweep_params = {}
                if "learning_rate" in wandb.config:
                    training_args.learning_rate = float(wandb.config["learning_rate"])
                    sweep_params["learning_rate"] = training_args.learning_rate
                if "per_device_train_batch_size" in wandb.config:
                    training_args.per_device_train_batch_size = int(wandb.config["per_device_train_batch_size"])
                    sweep_params["per_device_train_batch_size"] = training_args.per_device_train_batch_size
                if "weight_decay" in wandb.config:
                    training_args.weight_decay = float(wandb.config["weight_decay"])
                    sweep_params["weight_decay"] = training_args.weight_decay
                if "warmup_steps" in wandb.config:
                    training_args.warmup_steps = int(wandb.config["warmup_steps"])
                    sweep_params["warmup_steps"] = training_args.warmup_steps
                if "num_train_epochs" in wandb.config:
                    training_args.num_train_epochs = int(wandb.config["num_train_epochs"])
                    sweep_params["num_train_epochs"] = training_args.num_train_epochs
                if "gradient_accumulation_steps" in wandb.config:
                    training_args.gradient_accumulation_steps = int(wandb.config["gradient_accumulation_steps"])
                    sweep_params["gradient_accumulation_steps"] = training_args.gradient_accumulation_steps
                if "max_source_length" in wandb.config:
                    data_args.max_source_length = int(wandb.config["max_source_length"])
                    sweep_params["max_source_length"] = data_args.max_source_length

                if sweep_params:
                    logger.info("=" * 80)
                    logger.info("  W&B Sweep hyperparameters detected:")
                    for key, value in sweep_params.items():
                        logger.info(f"    {key}: {value}")
                    logger.info("=" * 80)

            logger.info(f"  Weights & Biases logging enabled")
            logger.info(f"  Project: {wandb.run.project}")
            logger.info(f"  Run: {wandb.run.name}")
            if wandb.run.entity:
                logger.info(f"  Entity: {wandb.run.entity}")

            if wandb.run.name:
                original_output_dir = training_args.output_dir
                training_args.output_dir = os.path.join(original_output_dir, wandb.run.name)
                logger.info(f"Output directory: {training_args.output_dir}")

    set_seed(training_args.seed)

    # ------------------------------ Load data ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = None
    if training_args.do_train:
        if rank == 0:
            logger.info("Loading training data...")
        train_data = load_data(data_args.train_data_path)
        train_dataset = build_tokenized_datasets(
            train_data,
            tokenizer,
            max_length=data_args.max_source_length,
            data_limit=train_size_wanted,
            num_proc=training_args.dataloader_num_workers,
        )
        if rank == 0:
            logger.info(f"Training data size: {len(train_dataset)}")

    eval_dataset = None
    eval_strategy = getattr(training_args, "eval_strategy", None)
    if training_args.do_eval or (
        training_args.do_train and eval_strategy is not None and str(eval_strategy) != "no"
    ):
        if rank == 0:
            logger.info("Loading validation data...")
        val_data = load_data(data_args.val_data_path)
        eval_dataset = build_tokenized_datasets(
            val_data,
            tokenizer,
            max_length=data_args.max_source_length,
            data_limit=val_size_wanted,
            num_proc=training_args.dataloader_num_workers,
        )
        if rank == 0:
            logger.info(f"Validation data size: {len(eval_dataset)}")

    # ------------------------------ Load model ------------------------------
    model_path = model_args.model_name_or_path
    if training_args.do_predict and not training_args.do_train:
        if os.path.exists(training_args.output_dir) and os.path.exists(
            os.path.join(training_args.output_dir, "pytorch_model.bin")
        ):
            model_path = training_args.output_dir
            if rank == 0:
                logger.info(f"Loading trained model from: {model_path}")
        elif os.path.exists(training_args.output_dir) and os.path.exists(
            os.path.join(training_args.output_dir, "model.safetensors")
        ):
            model_path = training_args.output_dir
            if rank == 0:
                logger.info(f"Loading trained model from: {model_path}")

    config = AutoConfig.from_pretrained(
        model_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if rank == 0:
        logger.info(f"Model will be trained on GPU: {torch.cuda.get_device_name(rank if world_size > 1 else 0)}")

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    # ------------------------------ Data collator ------------------------------
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
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
        metrics = compute_metrics(eval_preds, tokenizer)
        if "bleu" in metrics:
            if rank == 0:
                logger.info(f"BLEU: {metrics['bleu']:.4f}")
            if "wandb" in (training_args.report_to or []):
                import wandb
                if wandb.run is not None:
                    wandb.log(
                        {
                            "eval_bleu": metrics["bleu"],
                            "eval_gen_len": metrics.get("gen_len", 0),
                        }
                    )
        return metrics

    eval_strategy = getattr(training_args, "eval_strategy", None)
    if training_args.do_train and eval_strategy is not None and str(eval_strategy) != "no":
        if rank == 0:
            if training_args.predict_with_generate:
                logger.info("BLEU evaluation enabled")
            else:
                logger.warning(
                    "BLEU evaluation disabled - set 'predict_with_generate: true'"
                )

    # ------------------------------ Initialize trainer ------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper
        if training_args.predict_with_generate
        else None,
    )

    # ------------------------------ Training ------------------------------
    if training_args.do_train:
        if rank == 0:
            logger.info(
                f"Starting training. Checkpoints will be saved to: {training_args.output_dir}"
            )
            logger.info(f"  - Save every {training_args.save_steps} steps")
            logger.info(f"  - Keep last {training_args.save_total_limit} checkpoints")
            logger.info(
                f"  - Best model will be saved based on: {training_args.metric_for_best_model}"
            )

        train_result = trainer.train()

        trainer.save_model()
        if rank == 0:
            logger.info(f"Final model saved to: {training_args.output_dir}")

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # ------------------------------ Evaluation ------------------------------
    if training_args.do_eval:
        if rank == 0:
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
        if rank == 0:
            logger.info("Loading test data...")
        test_data = load_data(data_args.test_data_path)
        test_dataset = build_tokenized_datasets(
            test_data,
            tokenizer,
            max_length=data_args.max_source_length,
            data_limit=test_size_wanted,
            num_proc=training_args.dataloader_num_workers,
        )

        if rank == 0:
            logger.info("*** Predict ***")
        max_length = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else data_args.val_max_target_length
        )
        predict_results = trainer.predict(
            test_dataset, metric_key_prefix="predict", max_length=max_length
        )
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)

        if trainer.is_world_process_zero() and training_args.predict_with_generate:
            import numpy as np

            predictions = predict_results.predictions
            predictions = np.where(
                predictions != -100, predictions, tokenizer.pad_token_id
            )
            predictions = tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(
                training_args.output_dir, "generated_predictions.txt"
            )
            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                writer.write("\n".join(predictions))

    if world_size > 1:
        destroy_process_group()


def main():
    if os.getenv("RANK") is not None and os.getenv("WORLD_SIZE") is not None:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        config_path = sys.argv[1] if len(sys.argv) > 1 else None
        main_worker(rank, world_size, config_path)
    else:
        num_gpus = torch.cuda.device_count()
        world_size = num_gpus
        
        if world_size >= 2:
            config_path = sys.argv[1] if len(sys.argv) > 1 else None
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass
            
            mp.spawn(
                main_worker,
                args=(world_size, config_path),
                nprocs=world_size,
                join=True
            )
        else:
            config_path = sys.argv[1] if len(sys.argv) > 1 else None
            main_worker(0, 1, config_path)

if __name__ == "__main__":
    main()
