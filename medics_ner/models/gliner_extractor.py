from typing import Dict, List, Optional, Tuple

from gliner import GLiNER

from .extractor_base import EncoderSpanExtractor
from .span_dataclasses import NERSpan, NERSpans


class GLIEncoderSpanExtractor(EncoderSpanExtractor):
    def __init__(
        self,
        identifier: str,
        label_normalization_map: Optional[Dict] = None,
        threshold: float = 0.3,
    ):
        super().__init__(identifier, label_normalization_map)
        self.threshold = threshold

    def load_model(self, identifier=None):
        if identifier is None:
            identifier = self.identifier
        model = GLiNER.from_pretrained(identifier)
        model.eval()
        return model

    def _get_model_max_length(self) -> int:
        return self.model.config.max_len

    def get_tokens(self, text: str) -> List[str]:
        tokens_with_offsets = list(self.model.data_processor.words_splitter(text))
        return [token for token, start, end in tokens_with_offsets]

    def get_token_offsets(self, text: str) -> List[Tuple[int, int]]:
        tokens_with_offsets = list(self.model.data_processor.words_splitter(text))
        return [(start, end) for token, start, end in tokens_with_offsets]

    def _get_expected_sequence_length(self, text) -> int:
        return len(self.get_tokens(text)) + 2 * len(self.entities)

    def extract_spans_from_chunk(
        self,
        text: str,
        threshold: Optional[float] = None,
        label_normalization_map: Optional[Dict] = None,
    ) -> NERSpans:
        if not threshold:
            threshold = self.threshold

        if not label_normalization_map:
            assert self.label_normalization_map is not None, "label normalization MUST either be defined while initializing or while calling inference"
            label_normalization_map = self.label_normalization_map

        entities = list(set(label_normalization_map.values()))

        extracted_spans = self.model.predict_entities(
            text, entities, threshold=threshold
        )

        spans = []
        for entity in extracted_spans:
            normalized_label = label_normalization_map.get(entity["label"], None)

            if not normalized_label:
                continue

            spans.append(
                NERSpan(
                    start=entity["start"],
                    end=entity["end"],
                    span_text=entity["text"],
                    label=normalized_label,
                    confidence=entity["score"],
                )
            )

        return NERSpans(parent_text=text, spans=spans)


class GLITokenEncoderSpanExtractor(EncoderSpanExtractor):
    def __init__(
        self,
        identifier: str,
        entities: List,
        label_normalization_map: Optional[Dict] = None,
        threshold: float = 0.3,
    ):
        super().__init__(identifier, entities, label_normalization_map)
        self.threshold = threshold

    def load_model(self, identifier=None):
        if identifier is None:
            identifier = self.identifier
        model = GLiNER.from_pretrained(identifier, load_tokenizer=True)
        model.eval()
        return model

    def _get_model_max_length(self) -> int:
        return self.model.config.max_len

    def get_tokens(self, text: str) -> List[str]:
        tokens_with_offsets = list(self.model.data_processor.words_splitter(text))
        return [token for token, start, end in tokens_with_offsets]

    def get_token_offsets(self, text: str) -> List[Tuple[int, int]]:
        tokens_with_offsets = list(self.model.data_processor.words_splitter(text))
        return [(start, end) for token, start, end in tokens_with_offsets]

    def _get_expected_sequence_length(self, text) -> int:
        return len(self.get_tokens(text)) + 2 * len(self.entities)

    def extract_spans_from_chunk(
        self,
        text: str,
        entities: list = None,
        threshold: Optional[float] = None,
        label_normalization_map: Optional[Dict] = None,
    ) -> NERSpans:
        if not threshold:
            threshold = self.threshold

        if not label_normalization_map:
            label_normalization_map = self.label_normalization_map

        if entities is None:
            entities = self.entities

        if not label_normalization_map:
            label_normalization_map = {entity: entity for entity in entities}

        extracted_spans = self.model.predict_entities(
            text, entities, threshold=threshold
        )
        spans = []
        for entity in extracted_spans:
            normalized_label = label_normalization_map.get(entity["label"], None)

            if not normalized_label:
                continue

            spans.append(
                NERSpan(
                    start=entity["start"],
                    end=entity["end"],
                    span_text=entity["text"],
                    label=normalized_label,
                    confidence=entity["score"],
                )
            )

        return NERSpans(parent_text=text, spans=spans)
