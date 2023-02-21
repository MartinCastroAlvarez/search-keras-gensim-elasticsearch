"""
Search layer.
This file contains the interface with Elasticsearch.

References:
- https://www.elastic.co/guide/en/machine-learning/master/setup.html
"""

import os
import sys
import logging
from typing import Dict

import requests

import pandas as pd

from score import Score
from embedding import Embedding
from summary import Summary

logger: logging.RootLogger = logging.getLogger(__name__)


class SearchException(Exception):
    """
    Base exception for all the exceptions in this library.
    """


class FileException(SearchException):
    """
    Raised when trying to load a file that does not exist.
    """


class HostException(SearchException):
    """
    Raised if the hostname is invalid.
    """


class ProtocolException(SearchException):
    """
    Raised if the protocol is invalid.
    """


class PostException(SearchException):
    """
    Raised if the port is invalid.
    """


class IndexException(SearchException):
    """
    Raised if the index is invalid.
    """


class ModelException(SearchException):
    """
    Raised if the Score is invalid.
    """


class InjectionException(SearchException):
    """
    Raised if the injected model is not valid.
    """


class DataException(SearchException):
    """
    Raised if the loaded data is invalid.
    """


class AttributeException(SearchException):
    """
    Raised if an attribute is invalid.
    """


class ResponseException(SearchException):
    """
    Raised if the response from the cluster is invalid.
    """


class Search:
    """
    Application search layer.
    """

    # Name of the feature representing the doc embedding.
    EMBEDDINGS: str = 'embeddings'

    # Name of the feature representing the raw text.
    TEXT: str = 'text'

    # Name of the feature representing the doc embedding.
    TITLE: str = 'title'

    # Name of the id column in the dataset.
    ID: str = 'id'

    def __init__(self, index: str, host: str, port: int = 9200, protocol: str = 'https') -> None:
        """
        Interface constructor.
        """
        if not index or not isinstance(index, str):
            raise IndexException('Invalid index:', index)
        if not host or not isinstance(host, str):
            raise HostException('Invalid hostname:', host)
        if not protocol or not isinstance(protocol, str):
            raise ProtocolException('Invalid protocol:', protocol)
        if not port or not isinstance(port, int) or port <= 1024:
            raise HostException('Invalid port:', port)
        self.protocol: str = protocol
        self.index: str = index
        self.host: str = host
        self.port: int = port

    @property
    def url(self) -> str:
        """
        Search url builder.
        """
        return f'{self.protocol}://{self.host}:{self.port}'

    def __get(self, **kwargs) -> dict:
        """
        GET backend.
        """
        return self.__request(**kwargs, method='get')

    def __put(self, **kwargs) -> dict:
        """
        PUT backend.
        """
        return self.__request(**kwargs, method='put')

    def __post(self, **kwargs) -> dict:
        """
        POST backend.
        """
        return self.__request(**kwargs, method='post')

    def __request(self, method: str, uri: str, data: dict, ignore: list = None, **kwargs) -> dict:
        """
        PUT backend.
        """
        headers: dict = {
            'Content-Type': 'application/json',
        }
        uri: str = f'{self.url}/{uri}'
        logger.info('Search | URI: %s %s %s', method.upper(), uri, data)
        response: dict = requests.request(method, uri, headers=headers, json=data)
        logger.info('Search | Response[%s]: %s %s', response.status_code, response.reason, response.text)
        if response.status_code < 200 or response.status_code > 299:
            if not ignore or not any((
                keyword in response.text
                for keyword in ignore
            )):
                raise ResponseException('Invalid response:', response.status_code, response.reason)
        return response

    def init(self) -> None:
        """
        Creating the index mapping.
        """
        logger.info('Search | Initiliazing: %s', self.index)
        self.__put(
            uri=f'{self.index}/',
            data={
                "mappings": {
                    "properties": {
                        self.TITLE: {
                            "type": "text",
                        },
                        self.TEXT: {
                            "type": "text",
                        },
                        self.EMBEDDINGS: {
                            "type": "dense_vector",
                            "dims": Embedding.SIZE,
                        },
                    }
                }
            },
            ignore=[
                'resource_already_exists_exception',
            ]
        )

    def load(self, path: str, embedding: Embedding, summary: Summary) -> None:
        """
        Loading a list of records into Elasticsearch.
        """
        if not os.path.isfile(path):
            raise FileException('The dataset does not exist:', path)
        logger.info('Search | Loading: %s', path)
        df: pd.DataFrame = pd.read_csv(path)
        if self.ID not in df.columns:
            raise DataException("Missing id column:", self.ID)
        if self.TEXT not in df.columns:
            raise DataException("Missing text column:", self.TEXT)
        if self.TITLE not in df.columns:
            raise DataException("Missing title column:", self.TITLE)
        for index, row in df.iterrows():
            logger.info('Search | Loading[%s]: %s', index, row)
            self.save(
                doc_id=row[self.ID],
                text=row[self.TEXT],
                title=row[self.TITLE],
                summary=summary,
                embedding=embedding,
            )

    def save(self, doc_id: str, title: str, text: str, embedding: Embedding, summary: Summary) -> dict:
        """
        Indexing a new document into Elasticsearch.
        """
        logger.info('Search | Indexing: %s %s', doc_id, title)
        if not doc_id or not isinstance(doc_id, int):
            raise AttributeException('The document ID is invalid:', doc_id)
        if not title or not isinstance(title, str):
            raise AttributeException('The title is invalid:', title)
        if not text or not isinstance(text, str):
            raise AttributeException('The title is invalid:', title)
        if not isinstance(embedding, Embedding):
            raise InjectionException('Invalid embedding instance:', embedding)
        if not isinstance(summary, Summary):
            raise InjectionException('Invalid summary instance:', summary)
        return self.__post(
            uri=f'{self.index}/_doc/{doc_id}',
            data={
                self.TITLE: title,
                self.TEXT: text,
                self.EMBEDDINGS: embedding.get_embedding(summary.get_summary(text)).tolist(),
                # self.EMBEDDINGS: [
                #     embedding.tolist()
                #     for embedding in embedding.get_embeddings(summary.get_summary(text))
                # ]
            },
        )

    def search(self, query: str, score: Score, embedding: Embedding, size: int = 5) -> dict:
        """
        To get the top 10 documents whose embeddings are most similar to another one,
        you can use Elasticsearch's search API along with a script query that computes
        the cosine similarity between the embeddings of the query document and those
        of the indexed documents.
        """
        if not isinstance(embedding, Embedding):
            raise InjectionException('Invalid embedding instance:', embedding)
        if not isinstance(score, Score):
            raise InjectionException('Invalid scoring instance:', score)
        response: dict = self.__post(
            uri=f'{self.index}/_search',
            data={
                "size": size,
                "query": {
                    "function_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script_score": {
                            "script": {
                                "source": f"cosineSimilarity(params.queryVector, doc['{self.EMBEDDINGS}']) + 1.0",
                                "params": {
                                    "queryVector": embedding.get_embeddings(query).tolist()
                                }
                            }
                        }
                    }
                }
            }
        )
        logger.info('Search | Response: %s', response)
        documents_by_id: Dict[int, dict] = {}
        documents_by_relevance: Dict[int, float] = {}
        for hit in response['hits']['hits']:
            doc_id: int = hit['_source'][self.ID]
            text: str = hit['_source'][self.TEXT]
            relevance: float = score.predict(query=query, doc=text, embedding=embedding)
            documents_by_id[doc_id] = hit
            documents_by_relevance[doc_id] = relevance
        logger.info('Search | Scored: %s', documents_by_relevance)
        response['hits']['relevant_hits'] = [
            documents_by_id[doc_id]
            for doc_id in sorted(documents_by_relevance, key=documents_by_relevance.get, reverse=True)[:size]
        ]
        logger.info('Search | sorted: %s', response)
        return response


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    summary: Summary = Summary()
    summary.load('facebook/bart-base')
    embedding: Embedding = Embedding()
    embedding.load(path="glove.6B.300d.txt")
    search: Search = Search(index='my_es_index2', protocol='http', host='localhost', port=9300)
    search.init()
    search.load('data/search.csv', embedding=embedding, summary=summary)
    score: Score = Score()
    score.load('relevance.h5')
    print(search.search(query='green', score=score, embedding=embedding, size=5))
