"""
originally based on https://github.com/elsanns/xai-nlp-notebooks/blob/master/fine_tune_bart_summarization_two_langs.ipynb
"""

import torch
import numpy as np
import datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from tabulate import tabulate
import nltk
from datetime import datetime

from EntailmentReward import EntailmentReward
from constants import *

model_name = "facebook/bart-base"
trained_model_name = "results3/checkpoint-350000/"

if USE_TRAINED:
    model = AutoModelForSeq2SeqLM.from_pretrained(trained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(trained_model_name)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = datasets.load_dataset("xsum")

nltk.download("punkt", quiet=True)
metric = datasets.load_metric("rouge")

train_data_txt, validation_data_txt = dataset["train"], dataset["validation"]

if TOY:
    train_data_txt = dataset["train"].filter(lambda example, indice: indice <50, with_indices=True)
    validation_data_txt = dataset["validation"].filter(lambda example, indice: indice <16, with_indices=True)
    NUM_EPOCHS = 10
    WARMUP_STEPS = 50

if SUBSET:
    train_data_txt = dataset["train"].filter(lambda example, indice: indice < 25000, with_indices=True)
    validation_data_txt = dataset["validation"].filter(lambda example, indice: indice < 10000, with_indices=True)
    NUM_EPOCHS = 5

def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["document"], batch["summary"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,
)

validation_data = validation_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=validation_data_txt.column_names,
)



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

"""### Training arguments"""

training_args = Seq2SeqTrainingArguments(
    output_dir="results",
    num_train_epochs=NUM_EPOCHS,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=2, #32,  # demo
    per_device_eval_batch_size=2, #32,
    # learning_rate=3e-05,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=50,
    save_total_limit=3,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

if USE_RL:
    trainer = EntailmentReward(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
else:
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


#trainer.evaluate()

"""Train the model"""

if not EVAL_ONLY:
    trainer.train()

"""Evaluate after fine-tuning"""

trainer.evaluate()


"""## Evaluation

---

**Generate summaries from the fine-tuned model and compare them with those generated from the original, pre-trained one.**
"""

def generate_summary(test_samples, model):
    inputs = tokenizer(
        test_samples["document"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


model_before_tuning = AutoModelForSeq2SeqLM.from_pretrained(model_name)

test_samples = validation_data_txt.select(range(16))

summaries_before_tuning = generate_summary(test_samples, model_before_tuning)[1]
summaries_after_tuning = generate_summary(test_samples, model)[1]

print(
    tabulate(
        zip(
            range(len(summaries_after_tuning)),
            summaries_after_tuning,
            summaries_before_tuning,
        ),
        headers=["Id", "Summary after", "Summary before"],
    )
)
print("\nTarget summaries:\n")
print(
    tabulate(list(enumerate(test_samples["summary"])), headers=["Id", "Target summary"])
)
print("\nSource documents:\n")
print(tabulate(list(enumerate(test_samples["document"])), headers=["Id", "Document"]))
