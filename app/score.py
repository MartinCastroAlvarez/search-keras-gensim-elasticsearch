"""
Relevace Keras model.

References:
- https://radimrehurek.com/gensim/models/word2vec.html

A relevance model is a machine learning model that is used to estimate the relevance
of search results to a given query. The goal of a relevance model is to predict the
likelihood that a particular document is relevant to a user's query. This can be
achieved through supervised learning techniques, where the model is trained on a
dataset of query-document pairs, each labeled with a relevance score.
"""

import os
import base64
import json
import sys
import logging

from typing import Optional, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from keras.layers import Input, Dense, Dropout, Concatenate, Layer
from keras.models import Model

from embedding import Embedding

logger: logging.RootLogger = logging.getLogger(__name__)


class ScoreException(Exception):
    """
    Parent exception of all the exceptions raised in this library.
    """


class DatasetScoreException(ScoreException):
    """
    Raised when trying to train an invalid dataset.
    """


class ReferenceScoreException(ScoreException):
    """
    Raised when trying to reference an invalid model.
    """


class PredictionScoreException(ScoreException):
    """
    Raised when trying to predict invalid values.
    """


class ModelNotFoundException(ScoreException):
    """
    Raised when trying to load a non-existent model.
    """


class InjectionException(ScoreException):
    """
    Raised when trying to use an invalid Embedding instance.
    """


class Score:
    """
    A relevance model is a machine learning model that is used to estimate the relevance
    of search results to a given query. The goal of a relevance model is to predict the
    likelihood that a particular document is relevant to a user's query. This can be
    achieved through supervised learning techniques, where the model is trained on a
    dataset of query-document pairs, each labeled with a relevance score.
    """

    # The name of the column containing the documents feature.
    DOCUMENT: str = "doc"

    # The name of the column containing the queries feature.
    QUERY: str = "query"

    # The name of the column containing the target feature.
    TARGET: str = "relevance"

    # The batch size in a machine learning model refers to the number of training examples
    # used in one iteration of the model's optimizer algorithm. In other words, during the
    # training process, the model processes a certain number of samples at once and updates
    # its parameters based on the average gradient of the loss function over that batch of
    # samples. The batch size is usually a hyperparameter that can be tuned to balance between
    # faster training time and more accurate gradients.
    BATCH_SIZE: int = 32

    # In machine learning, the term "epoch" refers to a single pass through the entire
    # training dataset during the training phase. In other words, one epoch is completed
    # when every training example has been processed once by the model. The number of epochs
    # is a hyperparameter that determines the number of times the model will iterate over
    # the entire training dataset during the training phase. It is common to increase the
    # number of epochs in order to improve the model's accuracy, although doing so may also
    # increase the risk of overfitting. On the other hand, using too few epochs can result
    # in underfitting, where the model does not capture the patterns in the data adequately.
    EPOCHS: int = 3

    # In machine learning, it is common to split the available data into two or three subsets:
    # a training set, a validation set, and a test set. The size of the split can vary depending
    # on the size of the dataset, the complexity of the model, and other factors.
    SPLIT_SIZE: float = 0.1

    # A model optimizer is a tool used to optimize and improve the performance of machine
    # learning models. It takes a trained model and applies various techniques to optimize
    # the model for a specific hardware configuration, such as CPUs, GPUs, or other
    # specialized processing units. The optimization process involves several steps,
    # including reducing the size of the model, converting the model to an optimized
    # format that can be executed more efficiently on the target hardware, and tuning
    # the model's hyperparameters for better performance. The goal of the model optimizer
    # is to make the model more efficient and faster without compromising its accuracy
    # or quality. Common model optimizers include TensorFlow Lite Converter, ONNX Runtime,
    # and OpenVINO.
    OPTIMIZER: str = 'adam'

    # In machine learning, metrics are used to evaluate and quantify the performance of
    # a model. Metrics are numerical values calculated based on the true labels and the
    # predicted labels of a model. Metrics help us to understand how well the model is
    # performing and whether it is meeting the desired objective.
    METRICS: List[str] = ['accuracy', ]

    # In machine learning, the term "loss" refers to a function that measures how well a
    # model is performing with respect to the problem it is trying to solve. The loss function
    # takes the predicted outputs of the model and compares them to the true outputs, generating
    # a single scalar value that indicates how well the model is doing on the problem. The goal
    # of training a machine learning model is to minimize the value of the loss function, so that
    # the model can make accurate predictions on new data.
    #
    # Loss functions available:
    # - Mean Squared Error (MSE)
    # - Mean Absolute Error (MAE)
    # - Mean Absolute Percentage Error (MAPE)
    # - Mean Squared Logarithmic Error (MSLE)
    # - Binary Cross-Entropy
    # - Categorical Cross-Entropy
    # - Sparse Categorical Cross-Entropy
    # - Kullback-Leibler Divergence
    # - Hinge Loss
    # - Cosine Proximity
    LOSS_FUNCTION: str = 'binary_crossentropy'

    def __init__(self) -> None:
        """
        Model constructor.
        """
        self.model: Optional[Model] = None

    def __compile(self) -> None:
        """
        In Keras, a model is compiled before training. The compilation step configures
        the model for training by specifying the optimizer, loss function, and metrics.

        By compiling the model, you specify how it should be trained and evaluated,
        which enables the underlying TensorFlow engine to perform the necessary
        calculations efficiently.
        """
        # Accepting 2 inputs.
        query_input: Input = Input(shape=(Embedding.SIZE, ), dtype='float32')
        doc_input: Input = Input(shape=(Embedding.SIZE, ), dtype='float32')

        # Query branch.
        q: Layer = query_input
        q: Layer = Dense(2**9, activation='relu')(q)
        q: Layer = Dropout(0.1)(q)

        # Document branch.
        d: Layer = doc_input
        d: Layer = Dense(2**9, activation='relu')(d)
        d: Layer = Dropout(0.1)(d)

        # The concatenation layer is used to combine the outputs of two or more layers
        # by concatenating them along a specified axis. It takes a list of input tensors,
        # and outputs a single tensor that concatenates them along the specified axis,
        # allowing you to make predictions based on both the image and text inputs.
        x: Layer = Concatenate()([q, d])

        # Dense layers are the regular deeply connected neural network layer in Keras.
        # It is a type of layer that performs a linear operation on the input followed
        # by an activation function. Each neuron in a dense layer receives input from
        # all the neurons in the previous layer, and each neuron's output is passed to
        # every neuron in the next layer. Dense layers are commonly used in deep learning
        # models for various tasks such as image classification, natural language processing,
        # and recommendation systems.
        x: Layer = Dense(2**10, activation='relu')(x)
        x: Layer = Dense(2**8, activation='relu')(x)
        x: Layer = Dense(2**4, activation='relu')(x)

        # The sigmoid activation function on the output layer of the model will map
        # the output to the range of 0 to 1.
        x: Layer = Dense(1, activation='sigmoid')(x)

        # Creating the model.
        self.model: Model = Model(inputs=[query_input, doc_input], outputs=x)

        # Compiling the model.
        logger.info('Relevance | Compiling: %s', self.model)
        self.model.compile(
            optimizer=self.OPTIMIZER,
            loss=self.LOSS_FUNCTION,
            metrics=self.METRICS,
        )
        self.model.summary()

    def train(self, path: str, embedding: Embedding) -> None:
        """
        The training step of a model is the process of training or fitting the model to a dataset.
        This involves feeding the model with input data and the corresponding target output, and
        adjusting the model's internal parameters to minimize the difference between its predicted
        output and the true target output. This process is typically iterative and involves
        computing the error or loss between the model's predicted output and the true output
        for each input in the training dataset, and then updating the model's parameters in
        a direction that reduces the error. This is typically done using an optimization
        algorithm, such as gradient descent.
        """
        if not isinstance(embedding, Embedding):
            raise InjectionException('Invalid embedding:', embedding)
        if not path or not isinstance(path, str) or not os.path.isfile(path):
            raise DatasetScoreException('Invalid dataset path:', path)
        logger.info('Relevance | Training: %s', path)

        # Loading the dataset.
        logger.info('Relevance | Loading Dataset: %s', self.model)
        df: pd.DataFrame = pd.read_csv(path)
        if self.TARGET not in df.columns:
            raise DatasetScoreException("Missing target column:", self.TARGET)
        if self.DOCUMENT not in df.columns:
            raise DatasetScoreException("Missing document column:", self.DOCUMENT)
        if self.QUERY not in df.columns:
            raise DatasetScoreException("Missing query column:", self.QUERY)
        y: pd.DataFrame = df[self.TARGET].values

        # Getting the embeddings.
        logger.info('Relevance | Embedding: %s %s', df[self.DOCUMENT], df[self.QUERY])
        X_doc: np.array = np.array([
            e if (e := embedding.get_embedding(doc)).shape else np.zeros((Embedding.SIZE, ))
            for doc in df[self.DOCUMENT]
        ])
        X_query: np.array = np.array([
            e if (e := embedding.get_embedding(query)).shape else np.zeros((Embedding.SIZE, ))
            for query in df[self.QUERY]
        ])

        # Splitting the data into train and test sets.
        logger.info('Relevance | Splitting: %s %s %s', X_doc.shape, X_query.shape, y.shape)
        X_query_train, X_query_test, X_doc_train, X_doc_test, y_train, y_test = train_test_split(
            X_query, X_doc, y, test_size=self.SPLIT_SIZE
        )

        # Training the model.
        self.__compile()
        logger.info('Relevance | Training: %s', self.model)
        self.model.fit(
            x=[X_query_train, X_doc_train],
            y=y_train,
            validation_data=([X_query_test, X_doc_test], y_test),
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
        )

    def load(self, path: str) -> None:
        """
        Loading a pre-trained model from disk.
        """
        if not os.path.isfile(path):
            raise ModelNotFoundException('The relevance model does not exist.')
        logger.info('Releavance | Loading: %s', path)
        self.model: Model = load_model(path)

    def save(self, path: str) -> None:
        """
        Saving a trained model to disk.
        """
        if self.model is None:
            raise ReferenceScoreException('The relevance model has not been loaded.')
        logger.info('Releavance | Saving: %s', self.model)
        self.model.save(path, save_format='tf')

    def to_base64(self) -> str:
        """
        Base64 encoding is a method of encoding binary data in a format that can be transmitted
        or stored as ASCII text. It represents binary data in an ASCII string format by translating
        it into a radix-64 representation. This encoding is used for transmitting binary data over
        channels that are designed to handle textual data.
        """
        return base64.b64encode(json.dumps(self.model.to_json()).encode('utf-8')).decode('utf-8')

    def to_json(self) -> dict:
        """
        To use a Keras model with Elasticsearch Learning to Rank (LTR), you first need to
        convert the model to a format that can be used with LTR. Elasticsearch LTR requires
        a specific format called the "text" format, which is a representation of the model
        as a JSON string. This can be done using the to_json() method of the Keras model.
        """
        if self.model is None:
            raise ReferenceScoreException('The relevance model has not been loaded.')
        return self.model.to_json()

    def predict(self, doc: str, query: str, embedding: Embedding) -> float:
        """
        Using the pre-trained model to predict the relevance of the document before a given query.

        The predict() method is a function in machine learning models that is used to generate
        predictions for a given input based on the trained model. Once a machine learning model
        is trained, it can be used to make predictions on new data. The predict() method takes
        a set of input data and returns the predicted output of the model for that input.
        """
        if not isinstance(embedding, Embedding):
            raise InjectionException('Invalid embedding:', embedding)
        if self.model is None:
            raise ReferenceScoreException('The relevance model has not been loaded.')
        if not doc or not isinstance(doc, str):
            raise PredictionScoreException('Invalid document string:', doc)
        if not query or not isinstance(query, str):
            raise PredictionScoreException('Invalid query string:', query)
        logger.info('Releavance | Predicting: %s %s', query, doc)
        X_query: np.array = embedding.get_embedding(query)
        X_doc: np.array = embedding.get_embedding(doc)
        logger.info('Releavance | Predicting: %s %s', X_query.shape, X_doc.shape)
        predictions: List[dict] = self.model.predict(x=[
            X_query.reshape(1, Embedding.SIZE),
            X_doc.reshape(1, Embedding.SIZE),
        ])
        logger.info('Releavance | Predicted: %s', predictions)
        return predictions[0][0]


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    model: Score = Score()
    embedding: Embedding = Embedding()
    embedding.load(path="glove.6B.300d.txt")
    model.train('data/songs.csv', embedding=embedding)
    print(model.predict(query='green', doc='The grass was greener', embedding=embedding))
    model.save('relevance.h5')
