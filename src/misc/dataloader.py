import numpy as np
import pandas as pd

from src.misc.preprocessor import preprocessor

'''
to-do
세팅 읽어서 cached 된 파일 있는지 검사하고 있으면
status 관리해서 로드한 데이터가 tonizing이 필요한지 표시할 것
파일 이름 만드는 규칙 생각할 것
'''

class DataLoader():
    def __init__(self, cfg):
        self.train_data_path = cfg['data']['train_data_path']
        self.test_data_path = cfg['data']['test_data_path']
        self.cols = cfg['cols']
        self.need_tokenize = True
        
        train = pd.read_csv(self.train_data_path)
        test = pd.read_csv(self.test_data_path)

        self.X_train, self.Y_train = self.raw_loader(train, True)
        self.X_test, _ = self.raw_loader(test, False)

    def raw_loader(self, pd_raw, label=True):
        raw_data = [list(pd_raw[col]) for col in self.cols]

        X, Y = [], []
        if label:
            Y = np.array(pd_raw['label'])

        for i in range(len(raw_data[0])):
        # for i in range(1000):
            concat_sentence = ''
            for col in raw_data:
                if type(col[i]) is str:
                    concat_sentence += col[i] + ' '
            concat_sentence = preprocessor(concat_sentence)
            X.append(concat_sentence)

        return X, Y

    def load(self):
        return self.X_train, self.Y_train, self.X_test
