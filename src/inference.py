import lightning as pl
import torch
import transformers
from transformers import AdamW, T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer
from model import SummaryModel

tokenizer = T5Tokenizer.from_pretrained("t5-base")


checkpoint="base-checkpoint.ckpt"
model = SummaryModel()
model=model.load_from_checkpoint(checkpoint,map_location="cpu")


def generate_summary(question):

    inputs_encoding =  tokenizer(
        question,
        add_special_tokens=True,
        max_length= 512,
        padding = 'max_length',
        truncation='only_first',
        return_attention_mask=True,
        return_tensors="pt"
        )

    
    generate_ids = model.model.generate(
        input_ids = inputs_encoding["input_ids"],
        attention_mask = inputs_encoding["attention_mask"],
        max_length = 128,
        num_beams = 4,
        num_return_sequences = 1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        )

    preds = [
        tokenizer.decode(gen_id,
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True)
        for gen_id in generate_ids
    ]

    return "".join(preds)
