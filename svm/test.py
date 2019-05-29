# doc file va in ra du lieu duoc xu ly ban dau
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string


def preprocess(words):
    words = [word.lower() for word in words]
    words = [re.sub("</?.*?>", " <> ", word) for word in words]
    words = [re.sub("(\\d|\\W)+", " ", word) for word in words]
    words = [word.strip() for word in words]
    words = [str for str in words if str]
    return words


def sentence(line):
    words = line[0:len(line)].strip().split(" ")
    words = preprocess(words)
    return words

word ="fasdfds 12354 <fs>ef lll fsfsd> fds'fsdf"
print(sentence(word))
print(len('  '))
