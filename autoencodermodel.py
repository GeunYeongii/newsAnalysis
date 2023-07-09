from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class AutoEncoderModel:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

    def build_model(self):
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
        decoded = Dense(self.input_dim, activation='sigmoid')(encoded)
        autoencoder = Model(input_layer, decoded)
        return autoencoder
