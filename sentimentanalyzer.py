from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = pipeline('sentiment-analysis')

    def analyze(self, text):
        result = self.analyzer(text)[0]
        return result['score'] if result['label'] == 'POSITIVE' else -result['score']
