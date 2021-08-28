from tensorflow.keras.layers import Conv1D, BatchNormalization, Input, Embedding, Dense
from tensorflow.keras.layers import GlobalMaxPool1D, Concatenate
from tensorflow.keras.models import Model
from tcn import TCN

class CNNBuilder:
    def __init__(self, cfg):
        self.seq_len = cfg['seq_len']
        self.vocab_size = cfg['vocab_size']

        self.pass_num = cfg['model']['cnn']['pass_num']
        self.kernel_list = cfg['model']['cnn']['kernel_list']

    def conv_build(self, embedding, filter_num, kernel_size):
        layer = embedding
        for _ in range(self.pass_num):
            layer = Conv1D(filter_num, kernel_size=kernel_size, padding='same', activation='relu')(layer)
            layer = BatchNormalization()(layer)
        return layer

    def build(self):
        # Model Building
        inputs = Input(shape=(self.seq_len,))
        embedding = Embedding(self.vocab_size, 256, mask_zero=True, input_length=self.seq_len)(inputs)
        conv_list = [(self.conv_build(embedding, 128, kernel_size)) for kernel_size in self.kernel_list]
        concat = Concatenate()(conv_list)
        pooling = GlobalMaxPool1D()(concat)
        # outputs1 = Dense(15, activation='softmax', dtype='float32')(pooling)
        outputs2 = Dense(46, activation='softmax', dtype='float32')(pooling)

        # Train with validation
        # print('----------Train with validation started...----------')
        self.model = Model(inputs=inputs, outputs=outputs2)
        return self.model
