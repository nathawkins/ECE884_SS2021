import pandas as pd
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, ELMoEmbeddings, WordEmbeddings, TransformerWordEmbeddings, DocumentPoolEmbeddings
import numpy as np
import torch, flair
import os
import time

## Disable GPU
flair.device =  torch.device("cpu")

def create_embedding_from_text(input_text, embedding_method):
    sent = Sentence(input_text)
    document_embedding = DocumentPoolEmbeddings([embedding_method])
    document_embedding.embed(sent)
    return sent.embedding.detach().numpy()

def create_embedding_corpus(list_of_text, embedding_method, output_filename):
    embedding_array = []
    for input_text in list_of_text:
        embedding_array.append(create_embedding_from_text(input_text, embedding_method=embedding_method))
    np.save(output_filename, np.array(embedding_array))

if __name__ == "__main__":

    data_dir = "../data/"
    processed_text_w_labels = pd.read_csv(data_dir+"preprocessed_text_w_labels.csv", index_col = 0)

    titles = processed_text_w_labels.iloc[:,0].values
    bodies = processed_text_w_labels.iloc[:,1].values

    ## Create embeddings
    method_names     = ["flair-news-forward", "flair-news-backward", "flair-multi-forward", "flair-multi-backward", "elmo", "twitter", "glove", "fasttext", "bert"]
    method_functions = [FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast'), FlairEmbeddings("multi-forward"), FlairEmbeddings("multi-backward"), ELMoEmbeddings('original'), WordEmbeddings('en-twitter'), WordEmbeddings('glove'), WordEmbeddings('news'), TransformerWordEmbeddings('bert-large-uncased')]
    for name, func in zip(method_names, method_functions):
        if not os.path.exists(data_dir+"titles_"+name+".npy"):
            print("Working On", name, data_dir+"titles_"+name+".npy")
            create_embedding_corpus(titles, func, data_dir+"titles_"+name+".npy")
        if not os.path.exists(data_dir+"bodies_"+name+".npy"):
            print("Working On", name, data_dir+"bodies_"+name+".npy")
            create_embedding_corpus(titles, func, data_dir+"bodies_"+name+".npy")