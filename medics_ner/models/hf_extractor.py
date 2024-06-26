from typing import Dict, Optional, Tuple

from transformers import pipeline

from .extractor_base import DecoderSpanExtractor, EncoderSpanExtractor
from .span_dataclasses import NERSpan, NERSpans
from .utils import (
    get_all_matches,
    get_iob_stripped_entity_set,
    merge_spans_with_same_labels,
    parse_generated_list,
)


class HFEncoderSpanExtractor(EncoderSpanExtractor):
    def __init__(
        self,
        identifier: str,
        label_normalization_map: Optional[Dict] = None,
    ):
        super().__init__(identifier, label_normalization_map)
        self.model_entities = get_iob_stripped_entity_set(
            list(self.model.model.config.label2id.keys())
        )  # these are the relevant entities

        if label_normalization_map is None:
            self.label_normalization_map = {
                ent: ent.lower() for ent in self.model_entities
            }

    def load_model(self, identifier=None):
        if identifier is None:
            identifier = self.identifier

        hf_pipeline = pipeline(
            model=identifier,
            task="ner",
            aggregation_strategy="max",
            device=self.device,
        )
        hf_pipeline.model.eval()
        return hf_pipeline

    def _get_model_max_length(self):
        return self.model.model.config.max_position_embeddings

    def get_tokens(self, text: str) -> list[str]:
        return self.model.tokenizer.tokenize(text)

    def get_token_offsets(self, text: str) -> list[Tuple[int, int]]:
        return self.model.tokenizer(text, return_offsets_mapping=True)[
            "offset_mapping"
        ][1:-1]

    def _get_expected_sequence_length(self, text) -> int:
        return len(self.get_tokens(text)) + 2

    def extract_spans_from_chunk(
        self,
        text: str,
        aggregation_strategy: str = "max",
        label_normalization_map: Optional[Dict] = None,
    ) -> NERSpans:
        """

        Args
            text
            aggregation_strategy
        """
        if not label_normalization_map:
            label_normalization_map = self.label_normalization_map

        extracted_spans = self.model(text, aggregation_strategy=aggregation_strategy)
        spans = []

        for entity in extracted_spans:
            if entity["entity_group"] in ["0", "O"]:
                continue

            normalized_label = label_normalization_map.get(entity["entity_group"], None)

            if not normalized_label:
                continue

            spans.append(
                NERSpan(
                    start=entity["start"],
                    end=entity["end"],
                    span_text=entity["word"],
                    label=normalized_label,
                    confidence=entity["score"],
                )
            )

        return NERSpans(parent_text=text, spans=spans)


class HFDecoderSpanExtractor(DecoderSpanExtractor):
    def __init__(
        self,
        identifier: str,
        prompt_file_path: str = None,
        label_normalization_map=None,
    ):
        super().__init__(identifier, label_normalization_map)
        self.prompt_file_path = prompt_file_path
        self.prompt_template = self.load_prompt(self.prompt_file_path)

    def load_model(self, identifier=None):
        if identifier is None:
            identifier = self.identifier
        model = pipeline("text-generation", model=identifier, device=self.device)
        model.model.eval()
        return model

    def _get_model_max_length(self) -> int:
        return self.model.model.config.max_position_embeddings

    def get_tokens(self, text: str) -> list[str]:
        return self.model.tokenizer.tokenize(text)

    def get_token_offsets(self, text: str) -> list[tuple[int, int]]:
        return self.model.tokenizer(text, return_offsets_mapping=True)[
            "offset_mapping"
        ][1:]

    def _get_expected_sequence_length(self, text) -> int:
        # In the line below, disease is a dummy entity.
        # We have assumed that we'd run inference for one entity at a time
        return (
            len(self.get_tokens(self.create_model_input(text, "disease"))) + 300
        )  # expected generation tokens

    def create_model_input(self, text: str, entity: str) -> str:
        if self.prompt_template is None:
            self.prompt_template = self.load_prompt(self.prompt_file_path)

        return self.load_prompt_with_variables(
            self.prompt_template, text=text, entity=entity
        )

    def get_text_completion(self, text: str, entity: str):
        input_prompt = self.create_model_input(text, entity)
        max_length = len(self.get_tokens(input_prompt)) + 200  # for generatation tokens
        chat_completion = self.model(
            input_prompt, do_sample=False, max_length=max_length, return_full_text=False
        )
        return chat_completion

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
        generated_text = chat_completion_obj[0]["generated_text"]

        identified_span_texts = parse_generated_list(generated_text)

        if normalized_label is None:
            normalized_label = entity

        ner_spans = []
        for span_text in identified_span_texts:
            matches = get_all_matches(span_text, document_text)
            for span_start, span_end in matches:
                ner_spans.append(
                    NERSpan(
                        start=span_start,
                        end=span_end,
                        span_text=span_text,
                        label=normalized_label,
                        confidence=-1,
                    )
                )
        return ner_spans

    def extract_spans_from_chunk(
        self, text: str, label_normalization_map=None
    ) -> NERSpans:
        if label_normalization_map is None:
            label_normalization_map = self.label_normalization_map

        entities = list(set(label_normalization_map.keys()))

        identified_spans = []
        for entity in entities:
            generated_obj = self.get_text_completion(text, entity)
            normalized_label = label_normalization_map.get(entity, None)
            entity_ner_spans = self.parse_generated_output(
                generated_obj, text, entity, normalized_label=normalized_label
            )
            identified_spans.extend(entity_ner_spans)

        # here we need to sort and deduplicate(keeping the largest span for now)
        if identified_spans:
            identified_spans = sorted(
                identified_spans, key=lambda x: (x.start, -len(x.span_text))
            )
            identified_spans = merge_spans_with_same_labels(identified_spans)

        return NERSpans(parent_text=text, spans=identified_spans)
