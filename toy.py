import json
import os

import numpy as np
import pandas as pd
import tensorflow_addons as tfa

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from src.misc.dataloader import DataLoader
from src.model.cnn import CNNBuilder
from src.tokenizer.tokenizer import Tokenizer


class Toy:
    def __init__(self):
        self.cfg = self.load_config()
        self.batch_size = self.cfg['training']['batch_size']
        self.valid_ratio = self.cfg['training']['valid_ratio']
        self.epochs = self.cfg['training']['epochs']
    
    def load_config(self):
        with open('./config/config.json') as config_file:
            cfg = json.load(config_file)
        return cfg

    def train(self):
        f1_s = tfa.metrics.F1Score(num_classes=46, threshold=0.5, average='macro')
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy', f1_s])
        self.model.summary()
        es = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=2)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0005)

        self.model.fit(self.X_train, self.Y_train,
                       epochs=self.epochs, 
                       batch_size=self.batch_size,
                       validation_data=(self.X_valid, self.Y_valid), 
                       callbacks=[reduce_lr, es])

    def export_submission(self):
        os.makedirs('./results', exist_ok=True)
        sub = self.model.predict(self.X_test, batch_size=self.batch_size)
        sub = np.argmax(sub, axis=-1)

        # test_sub = pd.read_csv('./data/raw/test.csv')
        test_sub = pd.read_csv(self.cfg['data']['test_data_path'])
        final_sub = pd.DataFrame(test_sub['index'])
        final_sub.insert(column='label', loc=1, value=sub)
        final_sub.to_csv('./results/submission.csv', index=False)

    def run(self):
        dataloader = DataLoader(self.cfg)
        self.X_train, self.Y_train, self.X_test = dataloader.load()
        need_tokenize = dataloader.need_tokenize

        if need_tokenize:
            tokenizer = Tokenizer(self.cfg, self.X_train, self.Y_train, self.X_test)
            self.X_train, self.Y_train, self.X_test = tokenizer.tokenize()
            self.cfg['vocab_size'] = tokenizer.vocab_size

        if self.cfg['model']['type'] == 'cnn':
            self.model = CNNBuilder(self.cfg).build()

        self.X_train, self.X_valid, self.Y_train, self.Y_valid = \
            train_test_split(self.X_train, self.Y_train, test_size=self.valid_ratio, random_state=777)

        self.train()
        self.export_submission()

if __name__ == '__main__':
    toy = Toy()
    toy.run()