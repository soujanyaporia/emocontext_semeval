import gensim.models as gsm

e2v = gsm.KeyedVectors.load_word2vec_format('./emoji2vec.txt', binary=False)
word = '😂'
word = 'hello'
if word in e2v:
    happy_vector = e2v[word]  # Produces an embedding vector of length 300
    print(happy_vector)

import emoji


def extract_emojis(str):
    return [c for c in str if c in emoji.UNICODE_EMOJI]


print(extract_emojis('rude😭😭'))
