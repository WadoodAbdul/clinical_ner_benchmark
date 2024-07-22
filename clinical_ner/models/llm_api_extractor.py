from .extractor_base import DecoderSpanExtractor
from .api_decoder_models import APIDecoderModel



class LLMSpanExtractor(DecoderSpanExtractor):
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
        default_generation_params = {"temperature":0.0}
        self.generation_parameters = default_generation_params | generation_parameters

    def load_model(self, identifier):
        return APIDecoderModel.from_predefined(identifier)

    def _get_model_max_length(self) -> int:
        return self.model._get_model_max_length()

    def get_tokens(self, text: str) -> list[str]:
        return self.model.get_tokens(text)

    def get_token_offsets(self, text: str) -> list[tuple[int, int]]:
        return self.model.get_token_offsets(text)

    def get_inference_config(self) -> dict:
        return {
            "label_normalization_map" : self.label_normalization_map,
            "generation_parameters" : self.generation_parameters,
            "prompt_template_identifier" : self.prompt_template_identifier,
            "library_version" : None
        }
    
    def get_text_completion(self, text: str=None, entity: str=None):
        model_input = self.prompt_template.create_model_input(text=text, entity=entity)
        # self.generation_parameters['max_length'] = self.model_max_length
        # self.generation_parameters['max_length'] = self._get_expected_sequence_length(text) + 100
        generated_text = self.model(
            model_input,
            **self.generation_parameters
            )
        return generated_text


