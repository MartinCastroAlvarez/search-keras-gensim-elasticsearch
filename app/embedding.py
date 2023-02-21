"""
Utilities for vectorizing documents.

References:
- https://radimrehurek.com/gensim/models/word2vec.html
- https://github.com/stanfordnlp/GloVe

In natural language processing (NLP), a word embedding is a representation of a word.
The embedding is used in text analysis.

Typically, the representation is a real-valued vector that encodes the meaning of the
word in such a way that words that are closer in the vector space are expected to be
similar in meaning.

Word embeddings can be obtained using language modeling and feature learning techniques,
where words or phrases from the vocabulary are mapped to vectors of real numbers.
"""

import os
import sys
import logging
import shutil
import smart_open

from typing import Set, List, Tuple

from gensim.models import Word2Vec, KeyedVectors

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

import numpy as np

lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
stop_words: Set[str] = set(stopwords.words('english'))
logger: logging.RootLogger = logging.getLogger(__name__)


class EmbeddingException(Exception):
    """
    Parent exception of all the exceptions raised in this library.
    """


class WordEmbeddingException(EmbeddingException):
    """
    Raised when trying to transform something to vector that is not a valid string.
    """


class TokensEmbeddingException(EmbeddingException):
    """
    Raised when trying to transform something to vector that is not a valid list of tokens.
    """


class ModelEmbeddingException(EmbeddingException):
    """
    Raised when trying to transform a string to vector without a valid model.
    """


class Embedding:
    """
    In natural language processing, an embedding is a numerical representation of a word
    or phrase that captures its meaning or context in a way that can be used as input to
    a machine learning model.
    """

    # NLTK uses the Penn Treebank POS tagging system, which includes the following POS tags:
    #
    # - CC: coordinating conjunction
    # - CD: cardinal digit
    # - DT: determiner
    # - EX: existential there (like: "there is" ... think of it like "there exists")
    # - FW: foreign word
    # - IN: preposition/subordinating conjunction
    # - JJ: adjective (large)
    # - JJR: adjective, comparative (larger)
    # - JJS: adjective, superlative (largest)
    # - LS: list marker (1)
    # - MD: modal (could, will)
    # - NN: noun, singular (desk)
    # - NNS: noun plural (desks)
    # - NNP: proper noun, singular (Harrison)
    # - NNPS: proper noun, plural (Americans)
    # - PDT: predeterminer (all, both, half)
    # - POS: possessive ending (parent's)
    # - PRP: personal pronoun (I, he, she)
    # - PRP$: possessive pronoun (my, his, hers)
    # - RB: adverb (very, silently)
    # - RBR: adverb, comparative (better)
    # - RBS: adverb, superlative (best)
    # - RP: particle (about)
    # - TO: to (to go, to the store)
    # - UH: interjection (errm)
    # - VB: verb base form (take)
    # - VBD: verb past tense (took)
    # - VBG: verb gerund/present participle (taking)
    # - VBN: verb past participle (taken)
    # - VBP: verb non-3rd person present (take)
    # - VBZ: verb 3rd person present (takes)
    # - WDT: wh-determiner (which)
    # - WP: wh-pronoun (who)
    # - WP$: possessive wh-pronoun (whose)
    # - WRB: wh-adverb (where)
    VALID_POS: List[str] = [
        'cc',
        'cd',
        'fw',
        'jj',
        'jjr',
        'jjs',
        'nn',
        'nns',
        'nnp',
        'nnps',
        'rb',
        'rbr',
        'rbs',
        'vb',
        'vbd',
        'vbg',
        'vbn',
        'vbp',
        'vbz',
    ]

    # The word2vec algorithm, which is used for generating word embeddings,
    # allows you to choose the size of the vector representation for each word.
    # This is typically referred to as the "embedding dimension" or "vector dimension".
    #
    # Choosing a larger vector dimension can potentially capture more information about
    # each word, but can also increase the computational complexity and memory requirements
    # of the model. On the other hand, choosing a smaller vector dimension can reduce the
    # computational complexity, but may result in less accurate representations of the words.
    SIZE: int = 300

    # Specifically, the window parameter specifies the maximum distance between the current
    # word being processed and the context words used to predict it. The context words are
    # the words within the window on either side of the current word. For example, if the
    # window size is set to 5, the context words for a given word would include the 5 words
    # to the left of it and the 5 words to the right of it, for a total of 10 context words.
    #
    # Setting a larger window size can potentially capture more information about the context
    # of each word, but can also increase the computational complexity and memory requirements
    # of the model. On the other hand, setting a smaller window size can reduce the
    # computational complexity, but may result in less accurate representations of the words.
    #
    # The optimal window size may depend on the specific dataset and downstream task you are
    # working on, so it may be necessary to experiment with different window sizes to find
    # the best value for your use case. In practice, typical window sizes range from 5 to 10 words.
    WINDOW: int = 5

    # Specifically, the min_count parameter specifies the minimum number of times a word must
    # occur in the corpus to be included in the vocabulary. Words that occur fewer times than
    # the specified threshold are ignored and not used for training the word embeddings.
    #
    # Setting a higher value for min_count can filter out rare words that may not have enough
    # context to learn useful embeddings, but can also reduce the size of the vocabulary and
    # potentially lead to loss of information. On the other hand, setting a lower value for
    # min_count can include more words in the vocabulary, but may also include noisy and less
    # informative words.
    MIN_COUNT: int = 1

    # In the gensim implementation of the word2vec algorithm, the "workers" parameter is used
    # to specify the number of worker threads to use during training of the word embeddings.
    WORKERS: int = 4

    # In the context of training machine learning models, epochs refer to the number of times the
    # entire dataset is presented to the model during training. During each epoch, the model updates
    # its parameters based on the error (or loss) it has made in predicting the target output.
    EPOCHS: int = 10

    def __init__(self) -> None:
        """
        Model constructor.
        """
        self.model: Word2Vec = Word2Vec(
            window=self.WINDOW,
            min_count=self.MIN_COUNT,
            workers=self.WORKERS,
            vector_size=self.SIZE,
        )

    def load(self, path: str) -> None:
        """
        GloVe (Global Vectors for Word Representation) is a popular method for learning word embeddings
        that is similar to Word2Vec. Gensim provides a simple way to load and use pre-trained GloVe
        embeddings in your natural language processing tasks.

        References:
        - https://github.com/jroakes/glove-to-word2vec
        """
        logger.info('Embedding | Loading mode: %s', path)
        if not os.path.isfile(path):
            raise ModelEmbeddingException('Embedding model not found:', path)
        if path.endswith('txt'):
            self.model.wv = KeyedVectors.load_word2vec_format(
                path,
                binary=False,
                encoding='ISO-8859-1',
                no_header=True,
            )
        else:
            self.model: Word2Vec = Word2Vec.load(path)
        logger.info('Embedding | Loaded: %s', self.model)
        logger.info('Embedding | Fuzzy Search: %s', self.model.wv.most_similar(positive=['australia'], topn=10))
        logger.info('Embedding | Similarity: %s', self.model.wv.similarity('woman', 'man'))

    def __prepend_line(self, infile: str, outfile: str, line: int) -> None:
        """
        Function use to prepend lines using bash utilities in Linux.

        References:
        - https://github.com/jroakes/glove-to-word2vec
        """
        with open(infile, 'r') as old:
            with open(outfile, 'w') as new:
                new.write(str(line) + "\n")
                shutil.copyfileobj(old, new)

    def __get_lines(self, glove_file_name: str) -> Tuple[int]:
        """
        Return the number of vectors and dimensions in a file in GloVe format.

        References:
        - https://github.com/jroakes/glove-to-word2vec
        """
        with smart_open.smart_open(glove_file_name, 'r') as f:
            num_lines = sum(1 for line in f)
        with smart_open.smart_open(glove_file_name, 'r') as f:
            num_dims = len(f.readline().split()) - 1
        return num_lines, num_dims

    def train(self, path: str) -> None:
        """
        The training step of a Word2Vec model involves learning high-dimensional vector representations
        (or embeddings) for each word in a corpus of text. The basic idea behind Word2Vec is to use a
        shallow neural network to predict the probability of observing a particular word given its
        context (i.e., the words that occur around it). The neural network is trained using a process
        called stochastic gradient descent (SGD), which updates the weights of the network based on
        the error between the predicted probability and the actual probability.

        The build_vocab() method of a Gensim Word2Vec model does not replace existing embeddings.
        Instead, it adds new words to the vocabulary of the model and initializes their embeddings
        randomly. The existing embeddings for words already in the vocabulary are not affected.
        """
        if self.model is None:
            raise ModelEmbeddingException('The embedding model has not been loaded.')
        if not isinstance(path, str) or not os.path.isfile(path):
            raise ModelEmbeddingException('The new list of documents is invalid.')
        with open(path) as file_handler:
            new_tokenized_documents: List[List[str]] = [
                token
                for document in file_handler.readlines()
                for token in self.preprocess(document)
            ]
        logger.info('Embedding | Training: %s', new_tokenized_documents)
        self.model.build_vocab(new_tokenized_documents)
        self.model.train(
            new_tokenized_documents,
            total_examples=len(new_tokenized_documents),
            epochs=self.EPOCHS,
        )
        logger.info('Embedding | Trained: %s', self.model)

    def save(self, path: str) -> None:
        """
        Saving a trained model to disk.
        """
        if self.model is None:
            raise ModelEmbeddingException('The embedding model has not been loaded.')
        logger.info('Embedding | Saving: %s', self.model)
        self.model.save(path)

    def preprocess(self, text: str) -> List[str]:
        """
        Text preprocessing is the process of cleaning and transforming raw text data into a format that
        is suitable for natural language processing (NLP) tasks such as text classification, sentiment
        analysis, machine translation, and information retrieval. The goal of text preprocessing is to
        remove noise and irrelevant information from the text while retaining the useful information
        that is needed for the NLP task.
        """
        logger.info('Embedding | Pre-processing: %s', text)
        if not isinstance(text, str) or not text:
            raise WordEmbeddingException('Invalid text:', text)
        return self.__lemmatize(self.__tag(self.__tokenize(text)))

    def __tag(self, tokens: List[str]) -> List[str]:
        """
        Part-of-speech (POS) tagging is the process of labeling each word in a text with its corresponding
        part of speech, such as noun, verb, adjective, etc. POS tagging is a key preprocessing step in
        many natural language processing (NLP) tasks, including text classification, sentiment analysis,
        and named entity recognition.
        """
        if not isinstance(tokens, list):
            raise TokensEmbeddingException('Invalid tokens:', tokens)
        tagged_tokens: List[Tuple[str]] = pos_tag(tokens)
        logger.info('Embedding | Tagging: %s', tagged_tokens)
        tokens: List[str] = [
            token
            for token, tag in tagged_tokens
            if tag.lower() in self.VALID_POS
        ]
        logger.info('Embedding | Tagged: %s', tokens)
        return tokens

    def __lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatizing is the process of reducing words to their base or dictionary form,
        called a "lemma." In natural language processing (NLP), lemmatization is often
        performed as a preprocessing step to standardize words and reduce the number
        of unique forms of a word that a model needs to consider.
        """
        if not isinstance(tokens, list):
            raise TokensEmbeddingException('Invalid tokens:', tokens)
        tokens: List[str] = [
            lemmatizer.lemmatize(token)
            for token in tokens
        ]
        logger.info('Embedding | Lemmatized: %s', tokens)
        return tokens

    def __tokenize(self, text: str) -> List[str]:
        """
        In natural language processing (NLP), tokenization refers to the process of breaking down a text
        into individual words or tokens. This is an essential step in many NLP tasks, as most algorithms
        and models operate on individual words rather than on full sentences or paragraphs.
        """
        if not isinstance(text, str):
            raise WordEmbeddingException('Invalid text:', text)
        tokens: List[str] = [
            token
            for token in word_tokenize(text.lower())
            if token.isalpha()
            and token not in stop_words
        ]
        logger.info('Embedding | Tokenized: %s', tokens)
        return tokens

    def get_embedding(self, text: str) -> np.array:
        """
        Getting a single embedding out of a document.
        """
        return np.mean(self.get_embeddings(text))

    def get_embeddings(self, text: str) -> np.array:
        """
        Transforms a string into an array using word embeddings.

        References:
        - https://radimrehurek.com/gensim/models/word2vec.html
        """
        if self.model is None:
            raise ModelEmbeddingException('The embedding model has not been trained.')
        logger.info('Embedding | Vectorizing[%s]: %s', len(text), text)
        embeddings: List[np.array] = [
            self.model.wv[token]
            for token in self.preprocess(text)
            # If a word is not in the Word2Vec model, it is called an out-of-vocabulary (OOV) word.
            # This means that the word is not included in the vocabulary of the Word2Vec model, and
            # therefore there is no pre-trained embedding vector for the word. This can be a problem
            # for downstream natural language processing tasks that require an embedding vector for
            # every word in the input text. To handle OOV words, one approach is to replace them with
            # a special token that represents all unknown words. Another approach is to use techniques
            # such as subword embeddings or character-level embeddings that can handle unseen words by
            # breaking them down into smaller units.
            if token in self.model.wv
        ]
        if not embeddings:
            logger.warning('Embedding | Ignored: %s', text)
            embeddings: List[np.array] = [
                np.zeros(shape=(self.SIZE, )),
            ]
        logger.info('Embedding | Embeddings[%s]: %s', len(embeddings), embeddings)
        assert len(embeddings) > 0
        assert isinstance(embeddings, list)
        return embeddings


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    embedding: Embedding = Embedding()
    embedding.load(path="glove.6B.300d.txt")
    embedding.train("data/docs.txt")
    print(embedding.get_embeddings("Scientists have discovered a new species of dinosaur in the Amazon rainforest."))
    embedding.save('embedding.h5')
