# Fraud Detection and Stock Price Forecasting

## Introduction
The primary aim of this project is to delve into two significant areas. The first area focuses on the detection of credit card fraud through the application of various machine learning models and neural networks. The objective here is to methodically compare these models to identify which one offers the most effective performance in recognizing and preventing fraudulent transactions. This comparative analysis will not only highlight the strengths and weaknesses of each model but also help in choosing the best-suited model for deployment in real-world scenarios.

The second area of the project is centered on forecasting the price movements of the NASDAQ index using Recurrent Neural Networks (RNN), particularly Long Short-Term Memory (LSTM) networks. LSTMs are a specialized kind of RNNs renowned for their ability to capture long-term dependencies in time-series data, making them ideal for tasks such as financial forecasting. The project will involve developing a predictive model using LSTM networks and then evaluating its performance by comparing its predictions with the actual historical price data of the NASDAQ index.

The NASDAQ (National Association of Securities Dealers Automated Quotations) is one of the largest and most well-known stock exchanges in the United States, primarily associated with the technology sector. It includes a broad spectrum of more than 3000 companies, ranging from tech giants like Apple and Google to emerging startups. The NASDAQ index is often seen as a key indicator of the health and trends within the tech industry and the broader stock market. By accurately predicting the movements of this index, investors and analysts can gain valuable insights into market trends and make more informed investment decisions.

The results of this project are expected to contribute valuable insights into the effectiveness of advanced machine learning techniques in financial applications, ranging from fraud detection to market prediction. By comparing the outcomes of different models, this project aims to highlight the potential and limitations of current technological approaches in tackling complex issues in the financial domain.

## Dataset

### Fraud Detection Dataset
The dataset available on Kaggle, titled "Fraud Detection in Credit Card Transactions," is an essential resource for developing and testing credit card fraud detection systems (Yashpal, 2022). This dataset is comprised of transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred in two days, where 492 frauds are out of 284,807 transactions. The dataset is highly unbalanced, representing a typical scenario for real-world financial datasets, where fraudulent transactions are much rarer than legitimate ones.

Each transaction in the dataset is described by 31 features, most of which are numerical input variables resulting from a PCA transformation. This was done to protect sensitive information. The features are labeled V1, V2, ..., V28, with 'Time' and 'Amount' being the only features not transformed. 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset, while 'Amount' is the transaction amount. The response variable, 'Class', is binary, where 1 indicates a fraudulent transaction and 0 indicates a non-fraudulent one.

This anonymized and PCA-transformed dataset is widely used by researchers and practitioners to apply machine learning techniques, especially for training models to identify patterns that signify fraudulent activities effectively.

### NASDAQ Index Price Dataset
For the second part of our project, focusing on Recurrent Neural Networks (RNN), we will utilize the Yahoo Finance API to extract five years of closing price data for the NASDAQ index. This comprehensive dataset will enable us to develop and refine our predictive model using LSTM networks, aiming to accurately forecast future price movements based on historical trends.

## General Methodology (Fraud Detection Dataset)

For each classification algorithm that we have tested, we have used a similar approach. Some of the parameters of the algorithm are fixed to ensure no overfitting or quicker convergence while the other parameters are tuned for improved accuracy using GridSearch. Scikit Learn provides the GridSearch functionality that sets each possible combination of hyperparameters provided for the classifier, fits the model, tests it using the scoring metric specified after cross-validation, and helps us identify the best set of hyperparameters.

The metric used for our testing is F1 Score, which is the harmonic mean of recall and precision, to combat the model skewness. As the dataset has significantly more number of normal transactions, a model that just classifies each transaction as normal would have been able to achieve 99%+ accuracy.

Precision and Recall are metrics that represent the True Positive Rates with respect to amongst the actual positive class, how many were identified as positive class and amongst those identified as a positive class, how many were correctly identified. A harmonic mean ensures the metric to be between 0 and 1 and gives equal importance to both these characteristics.

As we have a typical binary classification problem at our hand, the various algorithms we can test it against would include:<br>
1. Logistic Regression
2. Support Vector Machines
3. Random Forest, an ensemble of Decision Trees
4. K-Nearest Neighbours.

We compared the best estimator from the GridSearch from each of these four algorithms to identify the best one of them all using the F1 Metric and a confusion matrix that provides us with the true positives and negatives as well as the false positives and negatives.

Next we will use Artificial Neural Networks (ANN) to enhance our credit card fraud detection capabilities. ANNs are inspired by the biological neural networks that constitute animal brains, and they are particularly effective for pattern recognition tasks due to their ability to learn and model non-linear and complex relationships between inputs and outputs. By using a multi-layered structure of neurons, ANNs can discern subtle patterns in data that might indicate fraudulent activities. We will design and train an ANN model, optimizing its architecture—number of layers, neurons per layer, activation functions, and more—to effectively identify fraudulent transactions amidst our highly unbalanced dataset. This approach aims to leverage the deep learning strengths of ANNs to improve the accuracy and reliability of our fraud detection system.

ANNs are composed of:

Input layer: Where we enter our data.

Hidden layers: Intermediate layers where information is processed.

Output layer: Where we obtain the prediction or desired result.

Weights in Artificial Neural Networks are parameters that determine the strength and direction of connections between neurons. Each connection between neurons has an associated weight that controls the contribution of the input neuron to the output neuron.

How the process works

The ANN works on the data based on the inputs we give it to finally make its predictions in the output layer. These predictions are compared to the actual results based on a cost function, which basically measures the differences between the predictions and the actual values. Once these calculations have finished, the information is sent back to the beginning of the neural network in a process called 'backpropagation' in which the weights of the connections between the neurons are adjusted and the process is carried out again in order to reduce the cost function. This process is performed several times until the cost function is 0 or as small as possible.

Advantages and Challenges:

Advantages: ANNs can learn complex patterns and adapt to new data. They can detect fraud quickly and efficiently.

Challenges: Require large amounts of data to train correctly. Additionally, they can be difficult to interpret, which can be a problem in regulatory environments. Deep learning tends to work better with large amounts of data than machine learning models.

Activation functions in artificial neural networks (ANNs) serve a crucial role in determining the output of each neuron. They introduce non-linearities to the model, enabling it to learn complex patterns in the data. ReLU (Rectified Linear Unit) is commonly used in hidden layers because it is computationally efficient and helps mitigate the vanishing gradient problem during training. ReLU simply outputs the input if it's positive, otherwise, it outputs zero. On the other hand, the sigmoid function is often employed in the output layer, especially for binary classification tasks like fraud detection, because it squashes the output between 0 and 1, effectively representing probabilities. This makes it suitable for predicting binary outcomes, where values closer to 1 indicate a higher probability of fraud, while values closer to 0 indicate the opposite.

## Conclusion (Fraud Detection Dataset)

In our project, we've leveraged various machine learning models alongside a deep learning architecture (Artificial Neural Network, ANN). Upon analyzing the results, we found that the most effective machine learning model was Random Forest, boasting an impressive F1 score of 85.02%. Comparatively, the ANN achieved an F1 score of 85.51%, positioning at the same level of Random Forest. As we can see, deep learning models tend to perform better than machine learning models if they are fed with the correct amount of data, but Random Forest still achieved almost the same performance.

Our final recommendation is that for fraud detection, the best models to use are ANN and Random Forest. In cases where we have extensive data available, we recommend using ANN, as deep learning models tend to yield better results than machine learning models in these scenarios. However, in cases where data availability is limited, we recommend utilizing machine learning models, particularly Random Forest or KNN, as they have shown to deliver superior results. Additionally, Random Forest, being a tree-based ensemble method, it might provide better interpretability while explaining what features are important which the ANN wouldn't be able to explain.

## General Methodology (NASDAQ Index Price Dataset)

In this analysis, we embark on constructing a robust predictive tool for the NASDAQ, utilizing historical data from the past five years. Our objective is to not only capture the intricacies of NASDAQ trends but also to project them into the future, specifically targeting price trends for March 2024.

To accomplish this, we employ a Recurrent Neural Network (RNN), a deep learning architecture that excels in modeling sequences and temporal patterns in time-series data such as stock market trends. More specifically, we utilize Long Short-Term Memory (LSTM) units within our RNN. LSTMs are designed to effectively handle long-term dependencies in sequential data, making them particularly suited for our purposes.

The choice of LSTMs over traditional models like ARIMA is motivated by their ability to capture non-linear relationships and adapt to complex patterns in data, which are essential qualities for modeling the dynamics of the stock market. By feeding the RNN a sequence of historical price data, including the closing prices and trading volumes, the network learns to discern temporal patterns that influence market behavior.

Throughout the training phase, the RNN adjusts its internal parameters to minimize the discrepancy between its predictions and the actual stock prices observed in the dataset. This optimization is achieved using algorithms such as gradient descent, focusing specifically on the closing price of the NASDAQ index over the last five years.

Our approach aims not only to forecast the overall direction of the NASDAQ for the upcoming month but also to enhance our understanding of how neural networks can be effectively applied to financial time series forecasting. This deep analysis of NASDAQ trends allows us to grasp the complex underlying interactions and subtle changes that might be easily overlooked by simpler analytical methods, thereby providing a more comprehensive prediction of future market behavior.

## Conclusion (NASDAQ Index Price Dataset)

In conclusion, it is important to emphasize that this type of model serves to predict the trend of future prices rather than the price itself. Based on the results, we believe the model performed well when compared to real data, especially considering that we only used a single input variable. Therefore, we recommend incorporating more input variables in the future to potentially achieve even better results.

# References

Yashpal 2022, Fraud Detection - Credit Card, Kaggle, https://www.kaggle.com/datasets/yashpaloswal/fraud-detection-credit-card

yfinance. (2024) finance pypi. https://pypi.org/project/yfinance/
