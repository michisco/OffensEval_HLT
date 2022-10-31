import emoji
import string
string.punctuation
import inflect
import csv
import re

from sklearn.preprocessing import LabelEncoder

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

from wordsegment import load, segment
import pandas as pd
import numpy as np


def load_dataset(dataset_path = 'OLID/',
                dataset = 'olid-training-v1.0.tsv',
                dataset_label = '',
                label = 'subtask_a'):
    
    '''Read data from dataset and return a dataframe with a list of labels'''
    X_raw = pd.read_csv(dataset_path + dataset, sep ="\t")
    if(label == "subtask_b" and dataset_label == ''):
        X_raw = X_raw.dropna(subset=['subtask_b'])
        X_raw.reset_index(drop=True, inplace=True)
    
    X = preProcessing(X_raw)
    
    le = LabelEncoder()
    if(dataset_label != ''):
        col_list = ["tweet", "label"]
        temp_y = pd.read_csv(dataset_path + dataset_label, sep =",", names=["tweet", "label"])
        y = le.fit_transform(temp_y['label'])
    else:
        y = le.fit_transform(X[label])
        
    return X, y

def remove_stopwords(text):
    """Remove stop words from list of tokenized words"""
    res = []
    for t in text:
        if t not in stopwords.words('english'):
            res.append(t)
    return res
    
def lemmatizer(text):
    """Lemmatize verbs in list of tokenized words"""
    wnl = WordNetLemmatizer()
    res = []
    for t in text:
        lemma = wnl.lemmatize(t, pos='v')
        res.append(lemma)
    return res

def emoji_translate(tweet):
    """Substituted the emojis with their corresponding text representation"""
    tweet = emoji.demojize(tweet)
    emojis = re.findall(r":(\w+)", tweet)
    for i in range(len(emojis)):
        emojis[i] = segment(emojis[i])
        emojis[i] = ' '.join(str(item) for item in emojis[i])
        tweet = re.sub(r":(\w+):", emojis[i]+' ', tweet, 1)
    return tweet
    
def replace_numbers(text):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    res = []
    for t in text:
        if t.isdigit():
            new_t = p.number_to_words(t)
            res.append(new_t)
        else:
            res.append(t)
    return res
    
def remove_punctuation(text):
    """Remove punctuation from text"""
    res = []
    for t in text:
        t = re.sub(r'[^\w\s]', '', t)
        if t != '':
            res.append(t)
    return res

def remove_duplicateUser(text):
    """Remove duplicate user tags """
    count_user = 0
    res = []
    for t in text:
        if t == "@user":
            if count_user > 0:
                t = ''
            else:
                count_user = count_user + 1
            
        if t != '':
                res.append(t)
    return res

def remove_hashtags(text):
    """Remove hashtag tags created by TextProcessor"""
    no_hashtag_start = re.compile('(\s*)<hashtag>(\s*)')
    no_hashtag_end = re.compile('(\s*)</hashtag>(\s*)')
    res = []
    for t in text:
        t = no_hashtag_start.sub('', t)
        t = no_hashtag_end.sub('', t)
        if t != '':
            res.append(t)
    return res
    
def normalize(text):
    text = remove_duplicateUser(text)
    text = remove_hashtags(text)
    text = remove_punctuation(text)
    text = replace_numbers(text)
    text = remove_stopwords(text)
    text = lemmatizer(text)
    return text

def preProcessing(df):
    no_user = re.compile('(\s*)@USER(\s*)')
    no_url = re.compile('(\s*)URL(\s*)')
    no_hashtag_start = re.compile('(\s*)<hashtag>(\s*)')
    no_hashtag_end = re.compile('(\s*)</hashtag>(\s*)')
    
    #read and parse the unigrams and bigrams data from disk.
    load()
    text_processor = TextPreProcessor (
        # terms that will be normalized
        normalize=[ 'email', 'percent', 'money', 'phone', 'time', 'date', 'number'] ,
        # terms that will be annotated
        annotate={"hashtag"} ,
        fix_html=True ,  # fix HTML tokens

        unpack_hashtags=True ,  # perform word segmentation on hashtags

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer ( lowercase=True ).tokenize,
        dicts = [ emoticons ]
    )

    #substitute URL
    df['tweet'] = df['tweet'].apply(lambda x: no_url.sub(' http ', x))
    #translate emoji in text
    df['tweet'] = df['tweet'].apply(lambda x: emoji_translate(x))
    df['tweet'] = df['tweet'].apply(lambda x: x.lower())
    
    i = 0
    for text in df['tweet']:
        df['tweet'][i] = str(" ".join(normalize(text_processor.pre_process_doc(text))))
        i = i + 1
    
    
    print(df['tweet'][9])
    return df