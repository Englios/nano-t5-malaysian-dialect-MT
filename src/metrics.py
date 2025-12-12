import numpy as np
import evaluate
from transformers import AutoTokenizer

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds, tokenizer: AutoTokenizer):
    """Compute BLEU score for translation task."""
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    metric = evaluate.load("sacrebleu")
    result_dict = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result_dict["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
