
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
from train import batch_tokenize_preprocess, compute_metrics, generate_summary
trained_model_name = "final_models/bart_baseline"



dataset = datasets.load_dataset("xsum")

nltk.download("punkt", quiet=True)
metric = datasets.load_metric("rouge")

test_data_txt = dataset["test"]

model = AutoModelForSeq2SeqLM.from_pretrained(trained_model_name)
tokenizer = AutoTokenizer.from_pretrained(trained_model_name)

test_data = test_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=test_data_txt.column_names,
)

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
        eval_dataset=test_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
else:
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

"""Evaluate after fine-tuning"""

trainer.evaluate()

test_samples = test_data_txt.select(range(100))
summaries_after_tuning = generate_summary(test_samples, model)[1]
for sum in summaries_after_tuning:
    print(sum)
    print("---------------------------------------")
