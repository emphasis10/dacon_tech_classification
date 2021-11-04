import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer as kerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
# from gensim.models import Word2Vec

class Tokenizer():
    def __init__(self, cfg, X_train, Y_train, X_test):
        self.tkn_type = cfg['tokenizer']['type']
        self.seq_len = cfg['seq_len']
        self.char_level = cfg['tokenizer']['keras']['char_level']
        self.cfg = cfg['tokenizer']
        self.tokenizer = None
    
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

    def keras_tokenizer(self):
        self.tokenizer = kerasTokenizer(char_level=self.char_level, oov_token='<UNK>')
        self.tokenizer.fit_on_texts(self.X_train)

        self.vocab_size = len(self.tokenizer.word_index) + 1

        self.X_train = self.tokenizer.texts_to_sequences(self.X_train)
        self.X_test = self.tokenizer.texts_to_sequences(self.X_test)

        len_dist = [len(i) for i in self.X_train]
        maxlen = np.max(len_dist)
        meanlen = np.mean(len_dist)
        stdlen = np.std(len_dist)
        
        print("------Tokenized length distribution------")
        print("Max: %d\nMean: %d\nStandard Deviation: %d\n1 sigma: %d" \
        % (maxlen, meanlen, stdlen, meanlen + stdlen))

        self.X_train = pad_sequences(self.X_train, padding='pre', maxlen=self.seq_len)
        self.X_test = pad_sequences(self.X_test, padding='pre', maxlen=self.seq_len)

        self.Y_train = to_categorical(self.Y_train)
        # Y_primary_train = to_categorical(Y_primary_train)
        return self.X_train, self.Y_train, self.X_test


    def spm_tokenizer(self, train_X, model_path=None):
        # if model_path is None:
        #     with open('train_data.txt', 'w', encoding='utf8') as f:
        #         f.write('\n'.join(train_X))

        #     templates= '--input={} \
        #                 --model_prefix=spm_train \
        #                 --model_type={} \
        #                 --vocab_size={} \
        #                 --pad_id=0 \
        #                 --unk_id=1 \
        #                 --bos_id=2 \
        #                 --eos_id=3'

        #     train_input_file = "./train_data.txt"
        #     model_type = 'bpe'
        #     vocab_size = 32000

        #     cmd = templates.format(train_input_file,
        #                     model_type,
        #                     vocab_size)

        #     spm.SentencePieceTrainer.Train(cmd)
        return

    def tokenize(self):
        if self.tkn_type == 'keras':
            return self.keras_tokenizer()
        else:
            return self.spm_tokenizer()
