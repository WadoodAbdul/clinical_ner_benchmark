from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import jinja2
import torch

from .span_dataclasses import NERSpans
from .utils import (
    ceiling_division,
    get_char_label_map,
    get_list_of_token_label_tuples,
    read_template,
    split_text_into_parts,
)


class GenericSpanExtractor(ABC):
    """
    Abstract base class for all classes that perform the function of getting ner spans from a piece of text
    """

    def set_attributes_for_dataset(self, **kwargs):
        """
        Sets the attributes that are required for inference, to align output with dataset requirements.
        This is used for evaluation
        """
        for attribute, value in kwargs.items():
            if not hasattr(self, attribute):
                print(f"the {attribute=} does not exist in class and will not be reset")
            setattr(self, attribute, value)

    @abstractmethod
    def extract_spans_from_chunk(text: str, **kwargs) -> NERSpans:
        """
        If you are inheriting from this class, this function should be implemented to handle span extraction from any sequence length
        """
        pass

    def __call__(self, text: str, **kwargs) -> NERSpans:
        return self.extract_spans_from_chunk(text, **kwargs)



class SpanExtractor(GenericSpanExtractor):
    """
    Abstract base class for all classes that perform the function of getting ner spans from a piece of text
    """

    def __init__(
        self, identifier: str, label_normalization_map: Optional[dict[str]] = None
    ):
        super().__init__()
        self.identifier = identifier

        self.label_normalization_map = label_normalization_map
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model(identifier)
        self.model_max_length = self._get_model_max_length()

    @classmethod
    def from_predefined(cls, identifier, **kwargs):
        from .model_loader import load_ner_processor

        return load_ner_processor(identifier, **kwargs)

    @abstractmethod
    def load_model(self, model_name_or_path):
        """returns the loaded model
        This is called by Span Extractor class and assigned to self.model"""
        pass

    @abstractmethod
    def _get_model_max_length(self) -> int:
        """returns the max length of the loaded model"""
        pass

    @abstractmethod
    def get_tokens(self, text: str) -> list[str]:
        """returns the list of tokens from the input text. Represents model's tokenization behaviour.
        But is mainly used for getting token level ner output from spans."""
        pass

    @abstractmethod
    def get_token_offsets(self, text: str) -> list[Tuple[int, int]]:
        """returns the list of tuples that represent the token offsets(start, end) in the text
        ex -
            text = "how are you?"
            tokens -> ["how", "are", "you?"]
            token_offset -> [(0,3), (4,7), (8,12)]
        """
        pass

    @abstractmethod
    def _get_expected_sequence_length(self, text) -> int:
        """returns the expected sequence length(in tokens) to be used by the model
        In the case of decoder models, this would include both the input and the output"""
        pass

    @abstractmethod
    def extract_spans_from_chunk(self, text: str) -> NERSpans:
        """returns a ner spans in from a piece of text without worrying about seq len
        This is the model specific implementation of inference"""
        pass

    def get_token_level_ner_output_from_spans(
        self, ner_spans: Union[list[Dict], NERSpans], parent_text: Optional[str] = None
    ) -> list[str]:
        """returns a list of tuples of token and label
        example - [('a', 'O'), ('cat', 'ANIMAL'), ('sits', 'O')]
        the tokens with no labels are assigned to 'O'

        Useful for token level evaluation!
        """
        if not parent_text:  # in the case where model prediction/NERSpans is passed - can be solved if dataclass is subscriptable
            parent_text = ner_spans.parent_text
            ner_spans = [span.__dict__ for span in ner_spans.spans]

        char_label_map = get_char_label_map(ner_spans)

        token_offsets = self.get_token_offsets(parent_text)
        tokens = self.get_tokens(parent_text)

        return get_list_of_token_label_tuples(tokens, token_offsets, char_label_map)

    def __call__(self, text: str, **kwargs) -> NERSpans:
        """
        Handles max_length limitation and call extract_spans.

        Note: The current implementation
            - Assumes uniform token distriution with sentence chars/words.
            If for some reason a specific sentence section has lot of tokens, this implementation might break.
                - Should I add a try except block specifying this?
            - Assumes non overlapping chunks and therefore does not deduplicate spans
        """
        expected_tokens_during_inference = self._get_expected_sequence_length(text)

        if expected_tokens_during_inference > self.model_max_length:
            number_of_chunks = ceiling_division(
                numerator=expected_tokens_during_inference,
                denominator=self.model_max_length,
            )

            chunks, _ = split_text_into_parts(text, number_of_chunks)

            previous_text_length = 0
            all_spans = []
            for chunk in chunks:
                spans = self.extract_spans_from_chunk(chunk, **kwargs).spans
                _ = [
                    span.update_offset(previous_text_length) for span in spans
                ]  # spans are updated in place

                previous_text_length += len(chunk)  # +1 removed with separator

                all_spans.extend(spans)

            parent_text = "".join(chunks)

            ner_spans = NERSpans(parent_text=parent_text, spans=all_spans)
        else:
            ner_spans = self.extract_spans_from_chunk(text, **kwargs)

        return ner_spans


class EncoderSpanExtractor(SpanExtractor):
    def __init__(self, identifier: str, label_normalization_map: dict[str] = None):
        super().__init__(identifier, label_normalization_map)


class DecoderSpanExtractor(SpanExtractor):
    def __init__(self, identifier: str, label_normalization_map=None):
        super().__init__(identifier, label_normalization_map)

    @staticmethod
    def load_prompt(prompt_template_file_path: str) -> str:
        """Returns the prompt with kwargs substititued in the string"""
        template = read_template(prompt_template_file_path)
        jinja_env = jinja2.Environment()
        return jinja_env.from_string(template)

    def load_prompt_with_variables(self, prompt_template, **kwargs) -> str:
        return prompt_template.render(kwargs)

    @abstractmethod
    def create_model_input(self, text: str, entity: str):
        """returns the model's inference input object.
        Used within the get_text_completion method. This output is used to call the model's inference method.
        """
        pass

    @abstractmethod
    def get_text_completion(self, text: str, entity: str):
        """returns the object that will be fed to parse_generated_output method to get ner spans"""
        pass

    @abstractmethod
    def parse_generated_output(
        self,
        chat_completion_obj,
        document_text: str,
        entity: str,
        normalized_label: str = None,
    ) -> NERSpans:
        """returns a list of ner spans of a single entity type from text
        let us assume we have only one beam
        ex - [{'generated_text': ' ["carcinoma"]'}]
        """
        pass
