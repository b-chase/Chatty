import re
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

normalizer = WordNetLemmatizer()

def get_part_of_speech(word):
  probable_part_of_speech = wordnet.synsets(word)
  pos_counts = Counter()
  pos_counts["n"] = len([item for item in probable_part_of_speech if item.pos()=="n"])
  pos_counts["v"] = len([item for item in probable_part_of_speech if item.pos()=="v"])
  pos_counts["a"] = len([item for item in probable_part_of_speech if item.pos()=="a"])
  pos_counts["r"] = len([item for item in probable_part_of_speech if item.pos()=="r"])
  most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
  return most_likely_part_of_speech

def preprocess_text(text, use_stop=False):
  stop_words = stopwords.words('english')
  #cleaned = re.sub(r'\W+', ' ', text).lower()
  cleaned = text.strip().lower()
  tokenized = word_tokenize(cleaned)
  if not use_stop:
    stop_words = []
  normalized = [normalizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized if token not in
                                                                                                  stop_words]
  return normalized