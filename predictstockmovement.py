import numpy as np

class PredictStockMovement:
    def __init__(self, model):
        self.model = model

    def predict(self, sentiment_scores):
        predictions = self.model.predict(np.array(sentiment_scores).reshape(-1, 1))
        return np.sign(predictions)
