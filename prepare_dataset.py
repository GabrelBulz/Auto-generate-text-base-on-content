import tensorflow as tf 
import re 
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

class DatasetParser:
    
    def __init__(self, path,  SIZE_BATCH):
        self.dataset = self.create_dataset(path)
        self.inp_tensor, self.inp_words_model, self.target_tensor, self.target_words_model = self.tokenize_dataset(self.dataset)

        #split dataset in 80% for train and 20 for validation
        self.inp_tensor_train, self.inp_tensor_val, self.target_tensor_train, self.target_tensor_val = train_test_split(self.inp_tensor, self.target_tensor, test_size=0.2)

        SIZE_INPUT = len(self.inp_tensor_train)
        self.dataset = tf.data.Dataset.from_tensor_slices((self.inp_tensor_train, self.target_tensor_train)).shuffle(SIZE_INPUT)
        self.dataset = self.dataset.batch(SIZE_BATCH, drop_remainder=True)

    def tokenize(self, text):
        lang_tokenizer  = tf.keras.preprocessing.text.Tokenizer(filters='')
        
        lang_tokenizer.fit_on_texts(text)
        tensor = lang_tokenizer.texts_to_sequences(text)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                            padding='post')

        return tensor, lang_tokenizer

    def tokenize_dataset(self, dataset):
        inp_sentences = dataset[0]
        target_sentences = dataset[1]
        inp_tensor, inp_words_model = self.tokenize(inp_sentences)
        target_tensor, target_words_model = self.tokenize(target_sentences)
        return inp_tensor, inp_words_model, target_tensor, target_words_model


    def clean_sentence(self, line):
        #get rid of punctuation
        line = re.sub(r"[,.;@#?!&$]+\ *", " ", line)
        line.strip()
        return line 


    def create_dataset(self, path):
        with open(path) as f:
            data = json.load(f)
            print(len(data))
            input_sentence = []
            target_sentence = []
            for pair in data:
                line1 = self.clean_sentence(pair["question1"])
                line2 = self.clean_sentence(pair["question"])
                input_sentence.append(line1)
                target_sentence.append(line2)
        return input_sentence, target_sentence


# """
#     Original dataset from Quora
#     https://figshare.com/s/5463afb24cba05629cdf -- 27/08/2020
#     the dataset contains pairs of questions, having the same meaning
# """
# PATH_TXT = "C:\\Users\\bulzg\\Desktop\\text_gen\\quora_raw_train.json"

# dataset_parser = DatasetParser(PATH_TXT, 64,100)
# print(len(dataset_parser.inp_tensor_train))

