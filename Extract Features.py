from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from string import punctuation
from gensim.models import keyedvectors
from nltk.stem.isri import ISRIStemmer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd
import gensim
import collections
import nltk
import re
import string



arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)


def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

    
def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text


def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)
    
    
def light_stem(text):
    words = text
    result = list()
    stemmer = ISRIStemmer()
    for word in words:
        word = stemmer.norm(word, num=1)      
        if not word in stemmer.stop_words:    
            word = stemmer.pre32(word)        
            word = stemmer.suf32(word)        
            word = stemmer.waw(word)          
            word = stemmer.norm(word, num=2)  
            result.append(word)
    return ' '.join(result)

def clean(text):
    text=str(text)
    translator = str.maketrans('', '', punctuations_list)
    text = text.translate(translator)
    text = remove_diacritics(text)
    text = remove_repeating_char(text)
    text=text.split()
    return light_stem(text)
    


model2=KeyedVectors.load_word2vec_format("../input/fasttext-pretrained-arabic-word-vectors/cc.ar.300.vec",limit=500000)



def transfer(text,model):
    ret=[]
    for w in text:
        if w in model:
            ret.append(model[w])
    if len(ret)==0:
        ret=np.zeros((300,))*00.1
    return ret
    
    
    
sent1='اهلا وسهلا'
sent1=clean(sent1)
sent1 = transfer(sent1,model2)
