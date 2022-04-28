import torch
import numpy as np
import ast
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from transformers import (get_linear_schedule_with_warmup,AdamW,AutoModel, AutoTokenizer, AutoModelForSequenceClassification)
from torch.utils.data import (TensorDataset,DataLoader, RandomSampler, SequentialSampler, Dataset)
from sklearn.utils import shuffle
from torch.utils.data import (TensorDataset,DataLoader,
                             RandomSampler, SequentialSampler, Dataset)
bertweet = AutoModel.from_pretrained('vinai/bertweet-base')
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', use_fast=False)

def prepare_model(model_class="vinai/bertweet-base",num_classes=512,model_to_load=None,total_steps=-1):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_class,
        num_labels = num_classes,
        output_attentions = False,
        output_hidden_states = True,
    )

    optimizer = AdamW(model.parameters(),
                    lr = 5e-5,
                    eps = 1e-8
                    )
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

    if model_to_load is not None:
        try:
            model.roberta.load_state_dict(torch.load(model_to_load))
            print("LOADED MODEL")
        except:
            pass
    return model, optimizer, scheduler


def load_data_lists(path):
    data_points_lists = []
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

        for line in lines:
            try:
                data_points_lists.append(ast.literal_eval(line))
            except:
                # Ignore lines with errors
                pass

    print('Found {} lines in "{}".'.format(len(lines), path))
    print('Successfully loaded {} data points from "{}".'.format(len(data_points_lists), path))

    return data_points_lists


COLUMN_NAMES = ['ID', 'Text', 'Sarcastic']

def construct_df(data_points_lists, column_names=COLUMN_NAMES):
    df = pd.DataFrame(data_points_lists, columns=column_names)
    df['ID'] = pd.to_numeric(df['ID'])
    df['Sarcastic'] = df['Sarcastic'].astype('bool')

    return df

train_df = construct_df(load_data_lists('data/text_data/train.txt'))
test_df = construct_df(load_data_lists('data/text_data/test.txt'), column_names=COLUMN_NAMES + ['Sarc_2'])
valid_df = construct_df(load_data_lists('data/text_data/valid.txt'), column_names=COLUMN_NAMES + ['Sarc_2'])

train_df = shuffle(train_df, random_state=42)
valid_df = shuffle(valid_df, random_state=42)
test_df = shuffle(test_df, random_state=42)

epochs = 10
total_steps = len(train_df) * epochs

model, optimizer, scheduler = prepare_model("vinai/bertweet-base" ,num_classes=10, model_to_load=None, total_steps = total_steps)


def bert_encode(df, tokenizer):
    input_ids = []
    attention_masks = []
    # print(df)
    for sent in df[['Text']].values:
        sent = sent.item()
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    inputs = {
        'input_word_ids': input_ids,
        'input_mask': attention_masks}

    return inputs


def prepare_dataloaders(train_df, test_df, batch_size=64):
    # Load the AutoTokenizer with a normalization mode if the input Tweet is raw

    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False, normalization=True)

    tweet_train = bert_encode(train_df, tokenizer)
    tweet_train_labels = train_df['Sarcastic'].astype(int)

    tweet_test = bert_encode(test_df, tokenizer)

    input_ids, attention_masks = tweet_train.values()
    labels = torch.tensor(tweet_train_labels.values)
    train_dataset = TensorDataset(input_ids, attention_masks, labels)

    input_ids, attention_masks = tweet_test.values()
    test_dataset = TensorDataset(input_ids, attention_masks)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size
    )
    return train_dataloader, test_dataloader


def predict(model, test_dataloader):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    preds = []

    for batch in test_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
            logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        for logit in logits:
            preds.append(logit)

    return np.array(preds)

if __name__ == "__main__":
    model_1 = torch.load('/content/drive/MyDrive/Colab Notebooks/berttweet_2epoch.pt')
