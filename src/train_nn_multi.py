import os
import traceback
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, multiprocessing
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from src.dataset import SummaryDataset
from src.embeddings import load_vocab, GloveEmbedding, load_embeddings, save_vocab, load_limited_embeddings, \
    GloveLimitedEmbedding
from src.model import AttentionModel, AttentionModelLimited
from datetime import datetime
import logging
import sys
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(filename="training_logs_multi.txt",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

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
    "vocab_path": "data/vocab_multi_500.json",
    "model_path": "saved_models/attention_summ_500.model",
    "train_data_path": "data/wikihow_final_clean_known_500_train.csv",
    "test_data_path": "data/wikihow_final_clean_known_500_test.csv",
    "batch_size": 16,
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

    for i, (text, summary) in enumerate(zip(data_f["text"], data_f["summary"])):
        if i % 5000 == 0:
            logging.info(f"Processed: {i}")
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
    for i, k in enumerate(bucket_data):
        print(f"Processing bucket: {i}")
        dataset = SummaryDataset(bucket_data[k]["text"], bucket_data[k]["summary"], vocab, text_len=k, summary_len=bucket_data[k]["max_summary_len"])
        bucket_dataset[k] = dataset

    return bucket_dataset

def get_bucket_dataloaders_wiki_multi_gpu(data_path, vocab, batch_size, rank, world_size):
    df = pd.read_csv(data_path, ",")
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=41)
    logging.info("Train Test Split complete.")

    train_bucket_datasets = get_bucketed_datasets(train_df, vocab)
    test_bucket_datasets = get_bucketed_datasets(test_df, vocab)

    train_bucket_loaders = {}
    for k in train_bucket_datasets:
        sampler = DistributedSampler(dataset=train_bucket_datasets[k], num_replicas=world_size, rank = rank)
        train_bucket_loaders[k] = DataLoader(dataset=train_bucket_datasets[k], shuffle=False, batch_size=batch_size, sampler = sampler, num_workers=0, pin_memory=True)
    test_bucket_loaders = {}
    for k in test_bucket_datasets:
        sampler = DistributedSampler(dataset=test_bucket_datasets[k], num_replicas=world_size, rank=rank)
        test_bucket_loaders[k] = DataLoader(dataset=test_bucket_datasets[k], shuffle=False, batch_size=batch_size, sampler = sampler, num_workers=0, pin_memory=True)

    return train_bucket_loaders, test_bucket_loaders

def convert_to_idx_word(wordtoidx):
    return {wordtoidx[word]: word for word in wordtoidx}

def generate_summary(model, loaders, idxtoword, rank = 0):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    model.eval()

    with torch.no_grad():
        for text_idxes, summary_idxes in list(loaders.values())[0]:
            for i in range(text_idxes.shape[0]):
                sel_t_idxes = text_idxes[i:i+1].to(device)
                sel_s_idxes = summary_idxes[i:i+1]
                all_summaries = model.process_with_beam_search(sel_t_idxes, idxtoword, beam_size=2)
                for gen_summary in all_summaries:
                    logging.info(gen_summary + "\n\n")
                break
            break

def print_summary(summaries, idxtoword): # batch, C, seq_len
    summary = summaries[0]
    idxes = summary.argmax(dim = -1)
    logging.info(" ".join([idxtoword[idx.item()] for idx in idxes]))

def get_test_loss(model, bucket_loader, loss_fn, rank = 0, idxtoword = None):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    model.eval()

    if idxtoword is not None:
        logging.info("Summary using BEAM SEARCH.")
        generate_summary(model, bucket_loader, rank, idxtoword)

    with torch.no_grad():
        total_batches = 0
        total_loss = 0
        for k in bucket_loader:
            loader = bucket_loader[k]
            for texts, summaries in loader:
                texts = texts.to(device)
                summaries = summaries.to(device)
                summary_pred = model(texts, summaries)
                if total_batches %50 == 0:
                    logging.info("\n\nTest data summary: \n")
                    print_summary(summary_pred, idxtoword)
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
    wordtoidx["<padding>"] = ln
    wordtoidx["<start>"] = ln + 1
    wordtoidx["<end>"] = ln + 2
    return wordtoidx

def train_nn_with_limited_embedding_multi_gpu(rank, world_size, train_options):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank)

    logging.info("INITIATED...")

    if train_options["load_vocab"]:
        wordtoidx = load_vocab(train_options["vocab_path"])
        embeddings = GloveLimitedEmbedding(len(wordtoidx.keys()), None, train_options["embedding_dim"])
    else:
        wordtoidx = get_vocab_from_dataset(train_options["train_data_path"])
        embedding_vec = load_limited_embeddings(wordtoidx, train_options["embedding_path"], train_options["embedding_dim"])
        embeddings = GloveLimitedEmbedding(len(wordtoidx.keys()), embedding_vec, train_options["embedding_dim"])
        save_vocab(wordtoidx, train_options["vocab_path"])

    idxtoword = convert_to_idx_word(wordtoidx)

    logging.info("Loaded Vocab")
    model = AttentionModelLimited(embeddings, len(wordtoidx.keys()))
    model.cuda(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    # if train_options["save_model"]:
    #     logging.info("SAVED MODEL")
        # save_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model

    logging.info("loading dataset")

    train_bucket_loaders, test_bucket_loaders = get_bucket_dataloaders_wiki_multi_gpu(train_options["train_data_path"], wordtoidx, train_options["batch_size"], rank, world_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=train_options["lr_rate"])

    min_test_loss = get_test_loss(model, test_bucket_loaders, loss_fn, rank, idxtoword)

    for epoch_num in range(train_options["epochs"]):
        logging.info(f"Epoch: {epoch_num}")
        total_train_batches = 0
        total_train_loss = 0
        grad_acc_batch = 4

        model.train()

        for k in train_bucket_loaders:
            train_loader = train_bucket_loaders[k]
            for batch_idx, (texts, summaries) in enumerate(train_loader):
                print(f"Batch sizes{texts.shape[0]}")
                texts = texts.cuda(rank)
                summaries = summaries.cuda(rank)
                summary_pred = model(texts, summaries)
                if total_train_batches % 100 == 0:
                    logging.info(f"Current time: {datetime.now()}")
                    logging.info(f"Batch num: {total_train_batches}, loss so far {total_train_loss / (1e-6 + total_train_batches)}")
                    logging.info("TRAINING SUMMARY\n\n")
                    print_summary(summary_pred, idxtoword)

                    logging.info("BEAM SEARCH SUMMARY: \n")
                    generate_summary(model, train_bucket_loaders,idxtoword, rank)
                loss = loss_fn(summary_pred[:, :-1].transpose(1,2), summaries[:, 1:])
                total_train_loss += loss.item()
                total_train_batches += 1

                loss = loss / grad_acc_batch

                loss.backward()
                if ((batch_idx + 1) % grad_acc_batch == 0) or (batch_idx == (len(train_loader) - 1)):
                    optimizer.step()
                    optimizer.zero_grad()

                del texts
                del summaries
                del summary_pred

        test_loss = get_test_loss(model, test_bucket_loaders, loss_fn, rank, idxtoword)

        logging.info(f"Train Loss: {total_train_loss / total_train_batches}")
        logging.info(f"Test Loss: {test_loss}")

        if min_test_loss > test_loss:
            min_test_loss = test_loss
            if train_options["save_model"]:
                save_model(model, train_options["model_path"])

    return model

def train_on_multi_gpu():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8110"
    world_size = torch.cuda.device_count()
    logging.info(f"{world_size} GPUS FOUND.")
    multiprocessing.spawn(train_nn_with_limited_embedding_multi_gpu, nprocs=world_size, args=(world_size, train_options, ))

#train_on_multi_gpu()

if __name__ == '__main__':
    train_on_multi_gpu()
#train_nn_with_limited_embedding(train_options)
