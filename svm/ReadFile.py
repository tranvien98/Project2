# doc file va in ra du lieu duoc xu ly ban dau
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string

import re


def preprocess(words):
    words = [word.lower() for word in words]
    words = [re.sub("</?.*?>", " <> ", word) for word in words]
    words = [re.sub("(\\d|\\W)+", " ", word) for word in words]
    words = [word.strip() for word in words]
    words = [word for word in words if len(word) > 1]
    words = [str for str in words if str]
    return words


fi = open("D:\\python\\preprocess\\stopwords.txt", "r")
if fi.mode == 'r':
    stopwords = []
    data = fi.readlines()
    for line in data:
      stopwords.append(line)
fi.close()

# ham xoa cac tu co trong stopword


def remove_stopwords(words):
    p_words = []
    rem = 0
    tu = 'all'
    for word in words:
        for tu in stopwords:

          if word.strip() == tu.strip():
              rem = 1
        if rem != 1:
          p_words.append(word)
        rem = 0
    words = p_words.copy()
    return words


def sentence(line):
    words = line[0:len(line)].strip().split(" ")
    words = preprocess(words)
    words = remove_stopwords(words)
    return words


def remove_metadata(lines):
    for i in range(len(lines)):
        if(lines[i] == '\n'):
            start = i+1
            break
    new_lines = lines[start:]
    return new_lines


def tokenize(path):
    f = open(path, "r")
    text_lines = f.readlines()
    doc_words = []

    for line in text_lines:
        words = sentence(line)
        doc_words.append(words)
    f.close()
    return doc_words


def flatten(list):
    new_list = []
    for i in list:
        for j in i:
            new_list.append(j)
    return new_list
def doi(num):
    w = 0
    if num == 2:
        w = 1
    if num == 14:
        w = 2
    if num == 19:
        w = 3
    return w
mypath = "D:\\python\\preprocess\\20_newsgroups"
folders = [f for f in listdir(mypath)]

files = []
for foldername in folders:
    folderpath = join(mypath, foldername)
    files.append([f for f in listdir(folderpath)])


pathname_list = []
for fo in range(len(folders)):
    for fi in files[fo]:
        pathname_list.append(join(mypath, join(folders[fo], fi)))

Y = []
for foldername in folders:
    folderpath = join(mypath, foldername)
    num_of_files = len(listdir(folderpath))
    for i in range(num_of_files):
        Y.append(foldername)

doc_train, doc_test, Y_train, Y_test = train_test_split(pathname_list, Y, random_state=0, test_size=0.25)
path = {}
for k in range(20):
    path[folders[k]] = k
fo = open("D:\\python\\svm\\preprocess_test.txt", "w+")
fil = open("D:\\python\\svm\\cate_test.txt", "w+")
numdoc = 0
tim = []
char = "<###>"
#maaa = ['0','2','14','19']
for file_path in doc_test:
    tim = file_path.strip().split('\\')
    line = []
    #if path[tim[4]] == 5:
   #    break
    fil.seek(0, 2)
    print(str(path[tim[4]]))
    fil.writelines(str(path[tim[4]])+'\n')
    line.append(str(path[tim[4]]))
    line.append(tim[5])
    words = flatten(tokenize(file_path))
    words = ' '.join(words)
    line.append(words)
    line = char.join(line)
    fo.seek(0, 2)
    fo.writelines(words+'\n')
    numdoc = numdoc + 1 
fil.close()
fo.close()
print(numdoc)


