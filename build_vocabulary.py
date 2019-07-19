import json
import nltk
import pickle
from collections import Counter
import pandas as pd

from config import train_file, valid_file, vocab_path, threshold
#nltk.download()

from vocab import Vocabulary

counter = Counter()  
data = pd.read_csv(valid_file,names=['id','text'])
cap = data["text"]

for cp in cap:  
    tokens = nltk.tokenize.word_tokenize(cp)
    counter.update(tokens)

words = [word for word, cnt in counter.items() if cnt >= threshold]

# Create a vocab wrapper and add some special tokens.
vocab = Vocabulary()
vocab.add_word('<pad>')
vocab.add_word('<start>')
vocab.add_word('<end>')
vocab.add_word('<unk>')

for i, word in enumerate(words):
    vocab.add_word(word)
with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)
print("Total vocabulary size: {}".format(len(vocab)))
print("Saved the vocabulary to '{}'".format(vocab_path))













