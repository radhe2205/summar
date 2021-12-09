import os
import traceback
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dataset import SummaryDataset
from src.embeddings import load_vocab, GloveEmbedding, load_embeddings, save_vocab, load_limited_embeddings, \
    GloveLimitedEmbedding
from src.model import AttentionModel, AttentionModelLimited
from datetime import datetime
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_options = {
    "embedding_dim": 300,
    "embedding_type": "glove",
    "embedding_path": "data/embeddings/glove822/glove.6B.{dims}d.txt",
    "base_path": "",
    "train_model": True,
    "load_model": False,
    "save_model": True,
    "load_vocab": False,
    "vocab_path": "data/vocab.json",
    "model_path": "saved_models/attention_summ.model",
    "train_data_path": "data/wikihow_known_500.csv",
    "test_data_path": "data/wikihow_known_500.csv",
    "batch_size": 32,
    "lr_rate": 0.0001,
    "epochs": 100,
}

def change_path_to_absolute(train_options):
    train_options["embedding_path"] = train_options["embedding_path"].format(dims = train_options["embedding_dim"])
    if train_options["base_path"] in train_options["model_path"]:
        return train_options

    for k in train_options:
        if not k.endswith("_path"):
            continue
        train_options[k] = train_options["base_path"] + train_options[k]
    return train_options

train_options = change_path_to_absolute(train_options)

def load_model(model, model_path):
    try:
        if not os.path.exists(model_path):
            return model

        model.load_state_dict(torch.load(model_path))
        return model
    except Exception as e:
        traceback.print_exc(e)
        logging.info("Error occured while loading, ignoring...")

def save_model(model, model_path):
    directory_path = "/".join(model_path.split("/")[:-1])
    if len(directory_path) > 0:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    torch.save(model.state_dict(), model_path)

def get_bucketed_datasets(data_f, vocab, bucket_width = 100):
    bucket_data = {}

    for text, summary in zip(data_f["text"], data_f["summary"]):
        text_word_c = len(text.split())
        bucket_key = (text_word_c // bucket_width) * bucket_width + bucket_width
        if bucket_key not in bucket_data:
            bucket_data[bucket_key] = {"text": [], "summary": [], "max_summary_len": 0}
        bucket_data[bucket_key]["text"].append(text)
        bucket_data[bucket_key]["summary"].append(summary)
        summary_word_c = len(summary.split())
        if bucket_data[bucket_key]["max_summary_len"] < summary_word_c:
            bucket_data[bucket_key]["max_summary_len"] = summary_word_c

    bucket_dataset = {}
    for k in bucket_data:
        dataset = SummaryDataset(bucket_data[k]["text"], bucket_data[k]["summary"], vocab, text_len=k, summary_len=bucket_data[k]["max_summary_len"])
        bucket_dataset[k] = dataset

    return bucket_dataset

def get_bucket_dataloaders_wiki(data_path, vocab, batch_size):
    df = pd.read_csv(data_path, ",")
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=41)

    train_bucket_datasets = get_bucketed_datasets(train_df, vocab)
    test_bucket_datasets = get_bucketed_datasets(test_df, vocab)

    train_bucket_loaders = {}
    for k in train_bucket_datasets:
        train_bucket_loaders[k] = DataLoader(dataset=train_bucket_datasets[k], shuffle=True, batch_size=batch_size)
    test_bucket_loaders = {}
    for k in test_bucket_datasets:
        test_bucket_loaders[k] = DataLoader(dataset=test_bucket_datasets[k], shuffle=False, batch_size=batch_size)

    return train_bucket_loaders, test_bucket_loaders

def get_test_loss(model, bucket_loader, loss_fn):
    model.eval()

    with torch.no_grad():
        total_batches = 0
        total_loss = 0
        for k in bucket_loader:
            loader = bucket_loader[k]
            for texts, summaries in loader:
                texts = texts.to(device)
                summaries = summaries.to(device)
                summary_pred = model(texts, summaries)
                loss = loss_fn(summary_pred[:, :-1].transpose(1,2), summaries[:, 1:])
                total_loss += loss.item()
                total_batches += 1
    return total_loss / total_batches

def get_vocab_from_dataset(data_path):
    all_words = set()
    all_words.add("<unk>")
    df = pd.read_csv(data_path, sep=",")
    for text, summary in zip(df["text"], df["summary"]):
        for word in text.split():
            if word in ("<start>", "<end>"):
                continue
            all_words.add(word)
        for word in summary.split():
            if word in ("<start>", "<end>"):
                continue
            all_words.add(word)
    wordtoidx = {word:i for i, word in enumerate(all_words)}
    ln = len(all_words)
    wordtoidx["<start>"] = ln
    wordtoidx["<end>"] = ln+1
    return wordtoidx

def train_nn_with_limited_embedding(train_options):
    if train_options["load_vocab"]:
        wordtoidx = load_vocab(train_options["vocab_path"])
        embeddings = GloveLimitedEmbedding(len(wordtoidx.keys()), None, train_options["embedding_dim"])
    else:
        wordtoidx = get_vocab_from_dataset(train_options["train_data_path"])
        embedding_vec = load_limited_embeddings(wordtoidx, train_options["embedding_path"], train_options["embedding_dim"])
        embeddings = GloveLimitedEmbedding(len(wordtoidx.keys()), embedding_vec, train_options["embedding_dim"])
        save_vocab(wordtoidx, train_options["vocab_path"])

    model = AttentionModelLimited(embeddings, len(wordtoidx.keys()))

    model.to(device)

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    if train_options["save_model"]:
        save_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model

    train_bucket_loaders, test_bucket_loaders = get_bucket_dataloaders_wiki(train_options["train_data_path"], wordtoidx, train_options["batch_size"])
    loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=train_options["lr_rate"])

    min_test_loss = 1e+6

    for epoch_num in range(train_options["epochs"]):
        logging.info(f"Epoch: {epoch_num}")
        total_train_batches = 0
        total_train_loss = 0
        grad_acc_batch = 4

        model.train()

        for k in train_bucket_loaders:
            train_loader = train_bucket_loaders[k]
            for batch_idx, (texts, summaries) in enumerate(train_loader):
                if total_train_batches % 100 == 0:
                    logging.info(f"Current time: {datetime.now()}")
                    logging.info(f"Batch num: {total_train_batches}, loss so far {total_train_loss / total_train_batches}")
                texts = texts.to(device)
                summaries = summaries.to(device)
                summary_pred = model(texts, summaries)
                loss = loss_fn(summary_pred[:, :-1].transpose(1,2), summaries[:, 1:])
                total_train_loss += loss.item()
                total_train_batches += 1

                loss = loss / grad_acc_batch

                loss.backward()
                if ((batch_idx + 1) % grad_acc_batch == 0) or (batch_idx == (len(train_loader) - 1)):
                    optimizer.step()
                    optimizer.zero_grad()

        test_loss = get_test_loss(model, test_bucket_loaders, loss_fn)

        logging.info(f"Train Loss: {total_train_loss / total_train_batches}")
        logging.info(f"Test Loss: {test_loss}")

        if min_test_loss > test_loss:
            min_test_loss = test_loss
            if train_options["save_model"]:
                save_model(model, train_options["model_path"])

    return model

def train_nn(train_options):
    if train_options["load_vocab"]:
        wordtoidx = load_vocab(train_options["vocab_path"])
        embeddings = GloveEmbedding(train_options["embedding_dim"], train_options["embedding_path"], False)
    else:
        embedding_vec, wordtoidx = load_embeddings(train_options["embedding_path"])
        embeddings = GloveEmbedding(train_options["embedding_dim"])
        with torch.no_grad():
            embeddings.embeddings.weight.data = embedding_vec
        save_vocab(wordtoidx, train_options["vocab_path"])

    model = AttentionModel(embeddings)

    model.to(device)

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    if train_options["save_model"]:
        save_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model

    train_bucket_loaders, test_bucket_loaders = get_bucket_dataloaders_wiki(train_options["train_data_path"], wordtoidx, train_options["batch_size"])
    loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=train_options["lr_rate"])

    min_test_loss = 20000

    for epoch_num in range(train_options["epochs"]):
        logging.info(f"Epoch: {epoch_num}")
        total_train_batches = 0
        total_train_loss = 0
        grad_acc_batch = 16

        model.train()

        for k in train_bucket_loaders:

            train_loader = train_bucket_loaders[k]
            for batch_idx, (texts, summaries) in enumerate(train_loader):
                if total_train_batches % 100 == 0:
                    logging.info(f"Current time: {datetime.now()}")
                    logging.info(f"Batch num: {total_train_batches}")

                texts = texts.to(device)
                summaries = summaries.to(device)
                summary_pred = model(texts, summaries)
                loss = loss_fn(summary_pred[:, :-1].transpose(1,2), summaries[:, 1:])
                total_train_loss += loss.item()
                total_train_batches += 1

                loss = loss / grad_acc_batch

                loss.backward()
                if ((batch_idx + 1) % grad_acc_batch == 0) or (batch_idx == (len(train_loader) - 1)):
                    optimizer.step()
                    optimizer.zero_grad()

        test_loss = get_test_loss(model, test_bucket_loaders, loss_fn)

        logging.info(f"Train Loss: {total_train_loss / total_train_batches}")
        logging.info(f"Test Loss: {test_loss}")

        if min_test_loss > test_loss:
            min_test_loss = test_loss
            if train_options["save_model"]:
                save_model(model, train_options["model_path"])

    return model

# model = train_nn(train_options)
model = train_nn_with_limited_embedding(train_options)
