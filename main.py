from dataloader import DataLoader
from sentimentanalyzer import SentimentAnalyzer
from autoencodermodel import AutoEncoderModel
from trainmodel import TrainModel
from predictstockmovement import PredictStockMovement

def main():
    # Load and preprocess data
    dataloader = DataLoader('news_data.xlsx')
    df = dataloader.load_data()

    # Perform sentiment analysis
    analyzer = SentimentAnalyzer()
    df['sentiment_score'] = df['CONTENT'].apply(analyzer.analyze)

    # Split data into training and testing sets
    x_train = df['sentiment_score'][:-100]
    x_test = df['sentiment_score'][-100:]

    # Build and train the autoencoder model
    autoencoder = AutoEncoderModel(input_dim=1, encoding_dim=32).build_model()
    trainer = TrainModel(autoencoder, x_train, x_test)
    trainer.train()

    # Predict stock price movement
    predictor = PredictStockMovement(autoencoder)
    df['predicted_movement'] = predictor.predict(df['sentiment_score'])

    print(df)

if __name__ == "__main__":
    main()
