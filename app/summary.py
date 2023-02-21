"""
Utilities for summarizing documents.

References:
- https://huggingface.co/docs/transformers/model_doc/bert

BartModel is a pre-trained transformer model that is part of the Hugging Face Transformers library,
which is built on top of TensorFlow. It is based on the BART (Bidirectional and Auto-Regressive Transformer)
architecture, which is a sequence-to-sequence model that can be used for tasks such as text generation, text
summarization, and machine translation. BartModel is capable of producing high-quality text summaries with
relatively few training examples, making it a useful tool for natural language processing tasks.
"""

import sys
import logging

from typing import List, Optional

from transformers import BartModel, TFBartForConditionalGeneration, BartTokenizer, BatchEncoding

logger: logging.RootLogger = logging.getLogger(__name__)


class SummaryException(Exception):
    """
    Parent exception of all the exceptions raised in this library.
    """


class WordEmbeddingException(SummaryException):
    """
    Raised when trying to transform something to vector that is not a valid string.
    """


class ModelSummaryException(SummaryException):
    """
    Raised when trying to load a non-existent file.
    """


class Summary:
    """
    BartModel is a pre-trained transformer model that is part of the Hugging Face Transformers library,
    which is built on top of TensorFlow. It is based on the BART (Bidirectional and Auto-Regressive Transformer)
    architecture, which is a sequence-to-sequence model that can be used for tasks such as text generation, text
    summarization, and machine translation. BartModel is capable of producing high-quality text summaries with
    relatively few training examples, making it a useful tool for natural language processing tasks.
    """

    # The lenght of the words in summarization result.
    LENGTH: int = 7

    def __init__(self) -> None:
        """
        Model constructor.
        """
        self.model: Optional[BartModel] = None
        self.tokenizer: Optional[BartTokenizer] = None

    def load(self, path: str) -> None:
        """
        Loading an existing Hugging Face model.
        """
        logger.info('Summary | Loading: %s', path)
        self.model: BartModel = TFBartForConditionalGeneration.from_pretrained(path)
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(path)
        logger.info('Summary | Loaded: %s', self.model)

    def save(self, path: str) -> None:
        """
        Saving a trained model to disk.
        """
        if self.model is None:
            raise ModelSummaryException('The embedding model has not been loaded.')
        logger.info('Summary | Saving: %s', self.model)
        self.model.save(path + '.model')
        self.tokenizer.save(path + '.tokenizer')

    def __get_tokens(self, text: str) -> BatchEncoding:
        """
        Vectorizing a text string.
        """
        logger.info('Summary | Vectorizing: %s', text)
        encoded_text: BatchEncoding = self.tokenizer.batch_encode_plus([text])
        logger.info('Summary | Encoded: %s', encoded_text)
        return encoded_text

    def get_summary(self, text: str) -> List[str]:
        """
        Method responsible for summarizing a text.
        """
        logger.info('Summary | Summarizing: %s', text)
        if self.model is None:
            raise ModelSummaryException('Model not loaded.')
        if self.tokenizer is None:
            raise ModelSummaryException('Tokenizer not loaded.')
        encoded_text: BatchEncoding = self.__get_tokens(text)
        summary_ids: 'EagerTensor' = self.model.generate(
            encoded_text['input_ids'],
            max_length=self.LENGTH,
            num_beams=4,
            early_stopping=True
        )
        logger.info('Summary | Summarizing: %s', summary_ids)
        summary: str = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logger.info('Summary | Summarized: %s', summary)
        return summary


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    summary: Summary = Summary()
    summary.load('facebook/bart-base')
    print(summary.get_summary("Scientists have discovered a new species of dinosaur in the Amazon rainforest."))
