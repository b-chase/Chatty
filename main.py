# this will be a very dumb chatbot

import random, nltk, pickle
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from process_text import preprocess_text
import numpy as np
import pandas as pd

class chat_bot:
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        try:
            self.phrases = self.load_learning()
            print("[[Saved learning file loaded.]]")
        except:
            print("[[No saved learning file found. Creating a new one.]]")
            self.phrases = {x: Counter() for x in range(2, max_seq_length + 1)}
        return

    def get_ngram_count(self, tokens, n, max=None):
        text_ngrams = ngrams(tokens, n)
        text_frequencies = Counter(text_ngrams)
        if max:
            text_frequencies = text_frequencies.most_common(max)
        return text_frequencies

    def pre_load_text(self, text_file_name):
        with open(text_file_name) as f:
            tokens = preprocess_text(f.read(), use_stop=False)
            self.update_phrases(tokens)
        return

    def update_phrases(self, tokens):
        # updates counters in the phrases dictionary with phrases counted from the new phrase
        for x in self.phrases:
            self.phrases[x].update(self.get_ngram_count(tokens, x))
        return

    def _get_response_tail(self, tokens):
        return tokens[-self.max_seq_length + 1:]

    def save_sent_patterns(self, text):
        sentences = sent_tokenize()
        for sentence in sentences:
            words = word_tokenize(sentence)
            tagged = pos_tag(words)
            just_tags = [pair[1] for pair in tagged]
            if 'sent_structures' not in self.phrases:
                self.phrases['sent_structures'] = Counter()
            else:
                self.phrases['sent_structures'][tuple(just_tags)] += 1
        return

    def save_learning(self, filename="default.mnd"):
        with open(filename, 'wb') as savefile:
            pickle.dump(savefile, self.phrases)

    def load_learning(self, filename="default.mnd"):
        with open(filename, 'rb') as openfile:
            self.phrases = pickle.load(openfile)

    def generate_response(self, last_words):
        next_word = ""
        if type(last_words) is str:
            # for debugging
            processed = preprocess_text(last_words, use_stop=False)
            self.update_phrases(processed)
            last_words = self._get_response_tail(processed)
        response = []
        while (next_word not in ".?!" or 3 > len(response)) and len(response) < 60:
            #print(f"The matching list of words is now: {last_words}")
            next_word = self._choose_next_word(last_words)
            #print(f"The next word is: {next_word}")
            response.append(next_word)
            last_words = last_words[2:] + [next_word]
        return " ".join(response).replace(" .", ".").replace(" !", "!").replace(" ?", "?")

    def _choose_next_word(self, last_words):
        matching = {}
        choice = ""
        for x in reversed(sorted(self.phrases)):
            last_w_sub = tuple(last_words[-x + 1:])
            matching.update({ngram: np.log(x*count) for ngram, count in self.phrases[x].items() if ngram[:-1] ==
                             last_w_sub})
        choice = random.choices(list(matching.keys()), list(matching.values()))[-1][-1]
        return choice

    def converse(self):
        quit_words = ['quit', 'stop', 'bye', 'leave']
        out_message = "Greetings! It's very nice to meet you. Perhaps you can tell me a little bit about yourself?"
        loop_run = True
        while loop_run:
            bot_tokens = preprocess_text(out_message, use_stop=False)
            bot_tail = self._get_response_tail(bot_tokens)
            user_text = input(out_message+"\n>> ")
            for w in quit_words:
                if w in user_text.lower():
                    print("Okay. Thanks for chatting! Bye!")
                    loop_run = False
            self.save_learning()
            user_tokens = bot_tail + preprocess_text(user_text, use_stop=False)
            self.update_phrases(user_tokens)
            user_tail = self._get_response_tail(user_tokens)
            out_message = self.generate_response(user_tail)
        return


testbot = chat_bot(10)
#testbot.pre_load_text("gygax_interview.txt")
testbot.converse()

