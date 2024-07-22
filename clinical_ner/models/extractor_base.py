import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, Literal

import torch
from jinja2 import Environment, FileSystemLoader

from .span_dataclasses import NERSpan, NERSpans
from .output_parsers import output_parser_loader
from .prompt_templates import PromptTemplate
from .utils import (
    ceiling_division,
    get_char_label_map,
    get_list_of_token_label_tuples,
    merge_spans_with_same_labels,
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
    Abstract base class for all classes that perform the function of getting ner spans from a piece of text.
    This has the added functionality 
    - to accomodate models with a limited context length.
    - return token level output
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
    def get_inference_config(self) -> dict:
        """returns the parameters used for inference"""

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

        Limitations: The current implementation assumes
            - Uniform token distriution across sentence chars/words.
            If for some reason a specific sentence section has lot of tokens, this implementation might break.
                - Should I add a try except block specifying this?
            - Non overlapping chunks and therefore does not deduplicate spans
            - Assumes the majority of the expected_tokens_during_inference to come from the text length
            If for decoder models, the prompt is the major contributor of expected tokens, then implementation might break
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
    def __init__(
            self, 
            identifier: str, 
            label_normalization_map=None,
            prompt_template_identifier=None,
            decoder_model_type:Literal["base", "chat"]="base",
            # output_parsing_function_identifier: str = None,
            generation_parameters:dict=None,
            # loop_for_each_entity:bool=True
            ):
        super().__init__(identifier, label_normalization_map)
        self.prompt_template_identifier = prompt_template_identifier
        self.prompt_template = PromptTemplate.from_predefined(prompt_template_identifier) if prompt_template_identifier is not None else None
        # self.output_parsing_function_identifier = output_parsing_function_identifier
        # self.parse_generated_output = output_parser_loader(output_parsing_function_identifier) if output_parsing_function_identifier is not None else None
        self.decoder_model_type = decoder_model_type
        self.generation_parameters = generation_parameters
        # self.loop_for_each_entity = loop_for_each_entity
    
    def set_attributes_for_dataset(self, **kwargs):
        super().set_attributes_for_dataset(**kwargs)

        assert self.prompt_template_identifier is not None, "Can't perform inference without a prompt template"
        self.prompt_template = PromptTemplate.from_predefined(self.prompt_template_identifier)
        # assert self.output_parsing_function_identifier is not None, "Can't parse output without a parsing function"
        # self.parse_generated_output = output_parser_loader(self.output_parsing_function_identifier)

    def create_model_input(self, text: str=None, entity: str=None):
        """returns the model's inference input object.
        Used within the get_text_completion method. This output is used to call the model's inference method.
        """
        return self.prompt_template.create_model_input(text=text, entity=entity)

    @abstractmethod
    def get_text_completion(self, text: str, entity: str):
        """returns the object that will be fed to parse_generated_output method to get ner spans"""
        pass

    def _get_expected_sequence_length(self, text) -> int:
        # In the line below, disease is a dummy entity.
        # We have assumed that we'd run inference for one entity at a time
        assert self.prompt_template is not None, "Can't run inference without prompt template"
            
        expected_sequence_text = self.prompt_template.get_expected_sequence_length_text(text)
        return len(self.get_tokens(expected_sequence_text))  

    def extract_spans_from_chunk(
        self, text: str, label_normalization_map=None
    ) -> NERSpans:
        if label_normalization_map is None:
            label_normalization_map = self.label_normalization_map

        entities = list(set(label_normalization_map.keys()))

        identified_spans = []
        if self.prompt_template.loop_for_each_entity:
            for entity in entities:
                generated_text = self.get_text_completion(text, entity)
                normalized_label = label_normalization_map.get(entity, None)
                entity_ner_spans = self.prompt_template.parsing_function(
                    generated_text, text, entity, normalized_label=normalized_label
                )
                identified_spans.extend(entity_ner_spans)
        else:
            generated_text = self.get_text_completion(text)
            ner_spans = self.prompt_template.parsing_function(
                generated_text, text
            )
            for ner_span in ner_spans:
                if not label_normalization_map.get(ner_span.label, None):
                    continue
                identified_spans.append(NERSpan(
                    start=ner_span.start,
                    end=ner_span.end,
                    span_text=ner_span.span_text,
                    label=label_normalization_map[ner_span.label],
                    confidence=-1,
                ))

        # here we need to sort and deduplicate(keeping the largest span for now)
        if identified_spans:
            identified_spans = sorted(
                identified_spans, key=lambda x: (x.start, -len(x.span_text))
            )
            identified_spans = merge_spans_with_same_labels(identified_spans)

        return NERSpans(parent_text=text, spans=identified_spans)