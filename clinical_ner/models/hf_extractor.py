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
        aggregation_strategy:str = "max"
    ):
        self.aggregation_strategy = aggregation_strategy
        
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
            aggregation_strategy=self.aggregation_strategy,
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
    
    def get_inference_config(self) -> Dict:
        import transformers
        
        return {
            "label_normalization_map" : self.label_normalization_map,
            "aggregation_strategy": self.aggregation_strategy,
            "library_version" : transformers.__version__
        }

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
        label_normalization_map=None,
        prompt_template_identifier: str = None,
        generation_parameters:dict = {},
    ):
        super().__init__(identifier, 
                         label_normalization_map=label_normalization_map,
                         prompt_template_identifier=prompt_template_identifier,
                         generation_parameters=generation_parameters,
                         )
        default_generation_params = {"do_sample":False, 
                                     "return_full_text":False}
        self.generation_parameters = default_generation_params | generation_parameters


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
    
    def get_inference_config(self) -> Dict:
        import transformers
        return {
            "label_normalization_map" : self.label_normalization_map,
            "generation_parameters" : self.generation_parameters,
            "prompt_template_identifier" : self.prompt_template_identifier,
            "library_version" : transformers.__version__
        }

    def get_text_completion(self, text: str, entity: str):
        model_input = self.prompt_template.create_model_input(text=text, entity=entity)
        self.generation_parameters['max_length'] = self.model_max_length
        # self.generation_parameters['max_length'] = self._get_expected_sequence_length(text) + 100
        chat_completion = self.model(
            model_input,
            **self.generation_parameters
            )
        if self.prompt_template.decoder_model_type == "base":
            # we have num_return_sequences = 1
            generated_text = chat_completion[0]['generated_text']
            
        elif self.prompt_template.decoder_model_type == "chat":
            generated_text = chat_completion[0]['generated_text'][-1]['content']

        return generated_text

