from tensorflow.keras.optimizers import Adam

class TrainModel:
    def __init__(self, model, x_train, x_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test

    def train(self):
        self.model.compile(optimizer=Adam(), loss='binary_crossentropy')
        self.model.fit(self.x_train, self.x_train,
                       epochs=50,
                       batch_size=256,
                       shuffle=True,
                       validation_data=(self.x_test, self.x_test))
