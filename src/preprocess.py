import re

import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.embeddings import load_vocab, load_embeddings

def find_all_num(data):
    all_ch_c = len(data)
    i = 0
    all_nums = set()
    while i < all_ch_c:
        if data[i].isnumeric():
            num = data[i]
            i+=1
            while data[i].isnumeric():
                num = num + data[i]
                i+=1
            all_nums.add(num)
        i+=1
    print(all_nums)

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    text = remove_mention_url(text=text)
    text = remove_entities(text=text)
    text = remove_hastags(text=text)
    text = lowercase(text=text)
    text = remove_non_ascii(text=text)
    text = add_space_latin(text=text)
    text = apostrophe_handling(text=text)
    text = add_space_punc(text=text)
    # text = remove_numbers(text=text)
    #text = remove_stop(text=text, stop=stop) # NOT NEEDED
    text = reduce_words(text=text)
    #text = stem_words(text=text, lemmatizer=lemmatizer)

    text = text.split()
    text = [w for w in text if w != '']
    text = ' '.join(text)
    return text

def stem_words(text, lemmatizer):
    if len(text) == 0:
        return text
    for word in text.split():
        text = text.replace(word, lemmatizer.lemmatize(word))
    return text

def remove_mention_url(text):
    text = re.sub('@[A-Za-z0-9_]+', '', text)
    text = re.sub('URL', '', text)
    return text

def remove_entities(text):
    text = text.replace('&lt;', '')
    text = text.replace('&gt;', '')
    text = text.replace('&amp;', '')
    return text

def remove_hastags(text):
    text = re.sub('#[A-Za-z0-9_]+', '', text)
    return text

def lowercase(text):
    text = text.lower()
    return text

def remove_non_ascii(text):
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return str(text)

def add_space_latin(text):
    text = re.sub('([.()!"#$%&*+,-/:;<=>?@^_`{|}~])', '\\1', text)
    return text

def apostrophe_handling(text):
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i had",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    for word in text.split():
        if word in contractions:
            text = text.replace(word, contractions[word])
    return text

def add_space_punc(text):
    # pat = re.compile(r"[()!?.,:;&@#*$%^+=-]")
    pat = re.compile(r"([\[()!?.,:;&@#*$%><^\"\'+=/\\\-\]])")
    # text = re.sub('[()!?.,:;&@#*$%^+=-]', ' ', text)
    text = pat.sub(' \\1 ', text)
    return text

def remove_numbers(text):
    text = re.sub("\d+", '', text)
    return text

def remove_stop(text, stop):
    text = text.split()
    text = [w for w in text if w not in stop]
    text = ' '.join(text)
    return text

def reduce_words(text):
    def reduced_word(w):
        s = w[0]
        curr_char = w[0]
        curr_count = 1
        for c in w[1:]:
            if c == curr_char:
                curr_count += 1
            else:
                curr_char = c
                curr_count = 1

            if curr_count <= 2:
                s += c
            else:
                continue
        return s

    if len(text) == 0:
        return text
    text = reduced_word(w=text)
    return text

def read_wikihow_dataset(file_path):
    df = pd.read_csv(file_path)
    return df["text"].values, df["headline"].values

def save_cleaned_text(texts, summaries, file_path): # np_arrays
    cleaned_summaries = []
    cleaned_texts = []

    for i, (text, summary) in enumerate(zip(texts, summaries)):
        if type(text) == float or type(summary) == float:
            continue
        if i % 5000 == 0:
            print(f"Cleaned {i}")
        cleaned_summaries.append(clean_text(summary))
        cleaned_texts.append(clean_text(text))

    cleaned_frame = pd.DataFrame({"text": cleaned_texts, "summary": cleaned_summaries})
    if file_path is not None:
        cleaned_frame.to_csv(file_path, sep = ",")
    return cleaned_texts, cleaned_summaries

def find_all_with_known_words(texts, summaries, wordtoidx):
    known_texts = []
    known_summaries = []
    for i, (text, summary) in enumerate(zip(texts, summaries)):
        if type(summary) == float or type(text) == float:
            continue
        not_found = False
        for word in text.split():
            if word not in wordtoidx:
                not_found = True
                break
        for word in summary.split():
            if word not in wordtoidx:
                not_found = True
                break
        if not_found:
            continue
        known_texts.append(text)
        known_summaries.append(summary)

    return known_texts, known_summaries

def save_known_text_summary(texts, summaries, wordtoidx, save_path):
    print(f"Length before known word filter {len(texts)}")

    known_texts, known_summaries = find_all_with_known_words(texts, summaries, wordtoidx)

    print(f"Length After known word filter {len(known_summaries)}")

    df = pd.DataFrame({"text": known_texts, "summary": known_summaries})
    if save_path is not None:
        df.to_csv(save_path, sep= ",")
    return known_texts, known_summaries

def clean_wikihow():
    print("Reading started.")
    texts, summaries = read_wikihow_dataset("data/wikihow.csv")
    print("Reading complete.")
    save_cleaned_text(texts, summaries, "data/wikihow_clean.csv")
    print("Cleaning complete.")

def plot_word_count_stats(file_path):
    df = pd.read_csv(file_path)
    word_count = {}
    for text in df["text"]:
        cnt = len(text.split())
        if cnt not in word_count:
            word_count[cnt] = 0

        word_count[cnt] += 1

    k = []
    v = []
    for cnt in word_count:
        k.append(cnt)
        v.append(word_count[cnt])

    plt.scatter(k, v, alpha=0.3)
    plt.legend()
    plt.show()

def clip_summary_word_count(file_path, word_count, target_file_path):
    df = pd.read_csv(file_path)
    summaries = []
    texts = []
    for text, summary in zip(df["text"], df["summary"]):
        if len(summary.split()) > word_count:
            continue
        summaries.append(summary)
        texts.append(text)

    print(f"total exemplars after clipping: {len(texts)}")
    pd.DataFrame({"text": texts, "summary": summaries}).to_csv(target_file_path, sep = ",")

def add_start_end(file_path):
    df = pd.read_csv(file_path)
    texts = []
    summaries = []
    for text, summary in zip(df["text"], df["summary"]):
        if type(text) == float or type(summary) == float:
            continue
        summaries.append("<start> " + summary + " <end>")
        texts.append("<start> " + text + " <end>")
    pd.DataFrame({"text": texts, "summary": summaries}).to_csv(file_path, sep = ",")

def all_known_count(emb_path, data_path):
    _, vocab = load_embeddings(emb_path, 50)
    df = pd.read_csv(data_path, sep=",")
    sum_count = 0
    for text, summary in zip(df["text"], df["summary"]):
        not_found = False
        if type(text) == float or type(summary) == float:
            continue
        for word in text.split():
            if word not in vocab:
                not_found = True
        if not not_found:
            for word in summary.split():
                if word not in vocab:
                    not_found = True
        if not_found:
            sum_count += 1
    print(f"Total known reviews: {sum_count}")

def filter_with_word_count(texts, summaries, word_count_t, word_count_s):
    filtered_texts = []
    filtered_summaries = []
    for text, summary in zip(texts, summaries):
        ln = len(text.split())
        lns = len(text.split())
        if ln > word_count_t:
            continue
        if lns > word_count_s:
            continue
        filtered_summaries.append(summary)
        filtered_texts.append(text)
    return filtered_texts, filtered_summaries

def final_preprocessing():
    # clean data
    # find all with existing emb
    # add start end
    # save

    df = pd.read_csv("data/wikihow.csv")
    print("CLEANING...")
    texts, summaries = save_cleaned_text(df["text"], df["headline"], None)

    print("Filtering with word count")
    texts, summaries = filter_with_word_count(texts, summaries, 500, 50)

    print("LOADING EMBEDDING...")
    _, vocab = load_embeddings("data/embeddings/glove822/glove.6B.50d.txt", 50)
    print("FINDING ALL KNOWNS...")
    texts, summaries = save_known_text_summary(texts, summaries, vocab, save_path = None)

    df = pd.DataFrame({"text": texts, "summary": summaries})

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=41)

    train_df.to_csv("data/wikihow_final_clean_known_50_train.csv", sep = ",")
    test_df.to_csv("data/wikihow_final_clean_known_50_test.csv", sep = ",")

    print(f"total datapoints {len(texts)}")
    add_start_end("data/wikihow_final_clean_known_50_train.csv")
    add_start_end("data/wikihow_final_clean_known_50_test.csv")
    print("DONE.")

# final_preprocessing()

# all_known_count("data/embeddings/glove822/glove.6B.50d.txt", "data/wikihow_clean.csv")
# df = pd.read_csv("data/wikihow_final_clean_known.csv")

def plot_word_count_distribution():
    text_lens = {i:0 for i in range(13000)}
    summary_lens = {i:0 for i in range(4000)}

    df = pd.read_csv("data/wikihow.csv")
    print(len(df))
    # df = df[:1000]
    all_words = set()
    max_len_t = 0
    max_len_s = 0
    for text, summary in zip(df["text"], df["headline"]):
        if type(text) != str or type(summary) != str:
            continue

        words_t = text.split()
        words_s = summary.split()

        text_lens[len(words_t)] += 1
        summary_lens[len((words_s))] += 1

        for word in words_t:
            all_words.add(word)
        for word in words_s:
            all_words.add(word)
        if len(words_t) > max_len_t:
            max_len_t = len(words_t)
        if len(words_s) > max_len_s:
            max_len_s = len(words_s)

    del text_lens[0]
    del summary_lens[0]

    plt.plot(text_lens.values())
    plt.xlabel("Count words")
    plt.ylabel("Number of data points")
    plt.title("Text length distribution")
    # plt.legend()
    plt.show()

    plt.plot(summary_lens.values())
    plt.xlabel("Count words")
    plt.ylabel("Number of data points")
    plt.title("Summary length distribution")
    # plt.legend()
    plt.show()

    print(f"Total distinct words {len(all_words)}")
    print(f"Max text length {max_len_t}")
    print(f"Max summary length {max_len_s}")

def remove_non_string(texts, summaries):
    clean_texts = []
    clean_summaries = []

    for i, (text, summary) in enumerate(zip(texts, summaries)):
        if type(text) == float or pd.isnull(text) or text.isspace() or not text:
            continue
        clean_texts.append(text)
        clean_summaries.append(summary)

    return clean_texts, clean_summaries

def create_clean_data_sith():
    # clean data
    # find all with existing emb
    # add start end
    # save

    df = pd.read_csv("data/wikihow.csv")
    print("Sith CLEANING...")
    texts, summaries = save_cleaned_text(df["text"], df["headline"], None)
    summaries = [" " for _ in summaries]

    print("LOADING EMBEDDING...")
    _, vocab = load_embeddings("data/embeddings/glove822/glove.6B.50d.txt", 50)
    print("FINDING ALL KNOWNS...")
    texts, summaries = save_known_text_summary(texts, summaries, vocab, save_path = None)

    texts, summaries = remove_non_string(texts, summaries)

    df = pd.DataFrame({"text": texts, "summary": summaries})

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=41)

    train_df.to_csv("data/wikihow_final_clean_known_train_sith.csv", sep = ",")
    test_df.to_csv("data/wikihow_final_clean_known_test_sith.csv", sep = ",")

    # print(f"total datapoints {len(texts)}")
    # add_start_end("data/wikihow_final_clean_known_train_sith.csv")
    # add_start_end("data/wikihow_final_clean_known_test_sith.csv")
    print("DONE.")

create_clean_data_sith()

# df = pd.read_csv("data/wikihow_known_500.csv")
# summ_cnt = {100: 0, 200:0, 300:0, 400:0, 500:0}
# max_text_cnt = 0
# all_words = set()
# for text, summary in zip(df["text"], df["summary"]):
#     ln = len(summary.split())
#     txt_ln = len(text.split())
#     for word in summary.split():
#         all_words.add(word)
#     for word in text.split():
#         all_words.add(word)
#     if txt_ln > max_text_cnt:
#         max_text_cnt = txt_ln
#
#     for ct in summ_cnt:
#         if ct > ln:
#             summ_cnt[ct] += 1
#
# print(f"Total words: {len(all_words)}")
# print(summ_cnt)
# print(max_text_cnt)
#
