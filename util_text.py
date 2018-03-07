import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def get_sentences(s):
    r = tokenizer.tokenize(s)
    return r