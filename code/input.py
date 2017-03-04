# imports
import pandas as pd
from pandas import Series,DataFrame

import numpy as np

# machine learning
from sklearn.feature_extraction.text import CountVectorizer

# preprocessing
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import string

# list input files
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# read data into dataframe
dataframes = {
    "cooking": pd.read_csv("../input/cooking.csv"),
    "crypto": pd.read_csv("../input/crypto.csv"),
    "robotics": pd.read_csv("../input/robotics.csv"),
    "biology": pd.read_csv("../input/biology.csv"),
    "travel": pd.read_csv("../input/travel.csv"),
    "diy": pd.read_csv("../input/diy.csv"),
}

test = pd.read_csv("../input/test.csv")


# remove html tags and uris from contents

uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

def stripTagsAndUris(x):
    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ""

print("removing html tags")
for df in dataframes.values():
    df["content"] = df["content"].map(stripTagsAndUris)


# construct a vocabulary and tokenize the content and tags

print("constructing corpus")
corpus = []
vectorizer = CountVectorizer(min_df=1)
for df in dataframes.values():
    for title in df['title']:
        corpus.append(title)
    for content in df['content']:
        corpus.append(content)
X = vectorizer.fit_transform(corpus)

analyzer = vectorizer.build_analyzer()
tokenizer = vectorizer.build_tokenizer()

print("tokenizing inputs")
for df in dataframes.values():
    df["title"] = df["title"].map(analyzer)
    df["content"] = df["content"].map(analyzer)
    df["tags"] = df["tags"].map(tokenizer)

print(dataframes['cooking'].head())
