# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:36:06 2020

@author: mishel
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:50:25 2020

@author: Mishel Elgawi 204563761, Elior mor 313168981
"""
import numpy as np
import pandas as pd
import argparse
import re 
import io
import logging
from datetime import datetime
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

handlers = [logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s', datefmt="%H:%M:%S", handlers=handlers)

#inital argparse parmaters 
parser = argparse.ArgumentParser(description="similarity.py --query <query file global path> --text <text file global path> --task <train/test> --data <training dataset in .csv format> --model <trained model>")
parser.add_argument('--query', type=str, default='aaa', help='file containing query sentence')
parser.add_argument('--text', type=str, default='aaa', help='file containing text sentence')
parser.add_argument('--task train', type=str, default='aaa', help='train the model')
parser.add_argument('--task test', type=str, default='aaa', help='test the model')
parser.add_argument('--data', type=str, default='aaa', help='training dataset in .csv format')
parser.add_argument('--model', type=str, default='aaa', help='trained model')
args = parser.parse_args()

def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'em", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"cannot", "can not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    return text

#read csv data set 
data = pd.read_csv(args.data)
data = data.dropna()

def preproceeing(data):
    #sepration to sentence and query 
    description = [sent for sent in data["description"]]
    tags = [tag for tag in data["tags"]]
    
    description = np.array(description)
    tags = np.array(tags)

    def toknization_and_stop_words(_list):
        tmp = []
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        for i in range(len(_list)):
            #tokenization the word
            token = tokenizer.tokenize(clean_text(_list[i]))
            # remove stopwords
            filtered_sentence = [w for w in token if not w in stopwords.words("english")]
            tmp.append(filtered_sentence)
        tmp = np.array(tmp)
        return tmp
        
    description = toknization_and_stop_words(description)
    tags = toknization_and_stop_words(tags)
    return description,tags

dictionry = {}  

#load the pretrain model of fasttext
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    print("Loading was successful")
    return data

#build vocab for the dataset
def build_vocab(_list):
    for i in _list:
        for j in i:
            if j not in dictionry:
                dictionry[j] = len(dictionry)

#embedding_matrix = {}
#
#def convert_words_to_vector(_list):
#    for key,value in dictionry.items():
#        embedding_vector = vector.get(key)
#        if embedding_vector is not None:
#            embedding_matrix[value] = embedding_vector


if __name__ == "__main__":
    startTime = datetime.now()
    logging.info("Script started")
    description, tags = preproceeing(data)
    logging.info("build vocab")
    build_vocab(description)
    build_vocab(tags)
    logging.info("loading fasttext vector")
    vector = load_vectors(r'fasttext_vector/wiki-news-300d-1M.vec')
    logging.info("done")
#    logging.info("convert_words_to_vector")
#    convert_words_to_vector(description)
#    convert_words_to_vector(tags)
#    logging.info("done")
    logging.info("Script ended, execution time: " + str(datetime.now() - startTime))