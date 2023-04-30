from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType, PeftConfig, PeftModel
import torch
from datasets import load_dataset
import os

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import datasets
from tydiqa2squad import load_tydiqa
import numpy as np

device = "cuda"
model_name_or_path = "google/mt5-base"



path = "tydiqa-goldp-v1.1-dev/tydiqa-goldp-dev-swahili.json"




tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)





def add_eos_to_examples(example):
    example['input_text'] = 'question: %s  context: %s </s>' % (example['question'], example['context'])
    example['target_text'] = '%s </s>' % example['answers']['text'][0]
    return example

def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=512)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=30)
    target_encodings[target_encodings == tokenizer.pad_token_id] = -100
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids'],
        'decoder_attention_mask': target_encodings['attention_mask']
    }
    return encodings

dataset = load_tydiqa(path)
dataset = dataset['train']
batch_size = 8
ids = dataset['id']
id2idx = {k:i for i, k in enumerate(ids)}
processed_datasets = dataset.map(
    add_eos_to_examples,
    load_from_cache_file = False
)
processed_datasets = processed_datasets.map(
    convert_to_features,
    batched = True,
    load_from_cache_file = False
)

eval_dataset = processed_datasets
columns = ['input_ids', 'labels', 'attention_mask', 'decoder_attention_mask']
eval_dataset.set_format(type='torch', columns=columns)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True, shuffle = False)




max_length = 128
batch_size = 8
ckpt = 1



peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=16)

peft_model_id = f"../{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model_id = peft_model_id
if(ckpt):
    try:
        print("loading checkpoint")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        model = get_peft_model(model, peft_config)
        checkpoint = torch.load('../checkpoints_tydiqa/checkpoint-12480/pytorch_model.bin')
        # print(checkpoint)
        model.load_state_dict(checkpoint)
    except Exception as E:
        print("error loading checkpoint")
        print(E)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        model = get_peft_model(model, peft_config)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)


model.to(device)

model.eval()
model.print_trainable_parameters()






eval_loss = 0
eval_preds = []
eval_labels = []
for step, batch in enumerate(tqdm(eval_dataloader)):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    loss = outputs.loss
    eval_loss += loss.detach().float()
    eval_preds.extend(tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True))
    eval_labels.extend(tokenizer.batch_decode(batch['labels'], skip_special_tokens=True))
eval_epoch_loss = eval_loss / len(eval_dataloader)
eval_ppl = torch.exp(eval_epoch_loss)
print("predicted: ", eval_preds[:25])
print("ground truth: ", eval_labels[:25])



predictions = [{'prediction_text':eval_preds[i], 'id':ids[i]} for i in range(len(ids))]
references = [{'answers': {'answer_start': dataset['answers'][i]['answer_start'], 'text': dataset['answers'][i]['text']}, 'id':ids[i]} for i in range(len(ids))]
squad_metric = datasets.load_metric("squad")
results = squad_metric.compute(predictions=predictions, references=references)
print(results)
