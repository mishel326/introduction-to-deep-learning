# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:32:02 2020

@author: Mishel Elgawi 204563761, Elior mor 313168981
"""
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
from wordcloud import WordCloud


background_color = "#101010"
height = 720
width = 1080

def read(name=r'sample.txt'):
    with open(name,encoding='utf-8') as file:
        data = file.read()
    return data

data = read()

tokenizer = RegexpTokenizer(r'\w+')
    
sentence = sent_tokenize(data)
tokenizetion = [tokenizer.tokenize(x.lower()) for x in sentence]
filtered_sentence = []
for x in tokenizetion:
    new_list = []
    for w in x:
        if w not in stopwords.words(): 
            new_list.append(w)
    filtered_sentence.append(new_list)
stemmer = PorterStemmer()
stem_sentence = []
for x in filtered_sentence:
    new_list = []
    for w in x:
        new_list.append(stemmer.stem(w))
    stem_sentence.append(new_list)
    
dictionry = {}

def build_dic(_list):
    for x in _list:
        for w in x:
            if w  not in dictionry:
                dictionry[w] = len(dictionry)

_data = dict()

build_dic(stem_sentence)

for word in stem_sentence:
    for w in word:
        _data[w] = _data.get(w,0) + 1
               
word_cloud = WordCloud(
    background_color=background_color,
    width=width,
    height=height,
    max_words=20
)
word_cloud.generate_from_frequencies(_data)
image = word_cloud.to_image()
image.show()
word_cloud.to_file('Mishel_Elior_cloud.png') 

for i in dictionry.keys():
    num = dictionry.get(i)
    dictionry[i] = np.zeros(len(dictionry))
    dictionry[i][num] = 1

def sentence_to_index(sentences):
    temp = []
    for sentence in sentences:
        temp.append(dictionry[sentence])
    return temp   
 
def list_to_index(_list):
    for i in range(len(_list)):
        _list[i] = sentence_to_index(_list[i])
            
list_to_index(stem_sentence)

with open(r'Mishel_Elior_1hot.txt','w',encoding='utf-8') as file:
    for i in stem_sentence:
        file.write('[')
        for j in i:
            file.write(str(j))
        file.write(']')
        file.write('\n')
