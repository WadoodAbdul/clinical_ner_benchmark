import os
import re
import logging
import requests
from abc import ABC, abstractmethod
from typing import Generator

from openai import OpenAI

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

class WhitespaceTokenSplitter:
    def __init__(self):
        self.whitespace_pattern = re.compile(r"\w+(?:[-_]\w+)*|\S")

    def __call__(self, text) -> Generator[str, int, int]:
        for match in self.whitespace_pattern.finditer(text):
            yield match.group(), match.start(), match.end()


class APIDecoderModel(ABC):
    """the interface that will be used by LLMSpanExtractor class"""
    api_key:str
    base_url:str

    def __init__(self,
                model_id:str,
                ) -> None:
        self.model_id = model_id
        self.base_url = self.base_url
        self.api_key = self.api_key
        # self.tokenizer = (
        #     WhitespaceTokenSplitter()
        # )  # if we find the need for diff tokenizer, a loading function will be added

    @classmethod
    def from_predefined(cls, model_id):
        if model_id in ["Meta-Llama-3-70b-Instruct", "Mixtral-8x7B-Instruct-v0.1", "Llama3-Med42-DPO-70B", "Llama3-Med42-70B"]:
            return LocalHostedDecoderModel(model_id)
        elif model_id in [ "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]:
            return OpenAIDecoderModel(model_id)
        raise NotImplementedError(f"APIDecoder class for {model_id=} has not been implemented or not added to the loader list")
        
    @abstractmethod
    def get_tokens(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def get_token_offsets(self, text: str) -> list[tuple[int, int]]:
        pass

    @abstractmethod
    def _get_model_max_length(self) -> int:
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        pass

    @abstractmethod
    def __call__(model_input, **generation_params):
        """calls the text generation method and returns the generated text."""
        pass


class LocalHostedDecoderModel(APIDecoderModel):
    api_key="EMPTY"
    base_url="http://dev-openai-api.med42.ai/v1"

    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)
        self.token_check_endpoint = self.base_url + "/token_check"
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url
            )
        self.tokenizer = (
            WhitespaceTokenSplitter()
        )
        self.headers = {"Content-Type": "application/json"}

    def get_tokens(self, text: str) -> list[str]:
        tokens_with_offsets = list(self.tokenizer(text))
        return [token for token, start, end in tokens_with_offsets]

    def get_token_offsets(self, text: str) -> list[tuple[int, int]]:
        tokens_with_offsets = list(self.tokenizer(text))
        return [(start, end) for token, start, end in tokens_with_offsets]

    def _get_model_max_length(self) -> int:
        data = {"prompts": [{"model": self.model_id, "prompt": "", "max_tokens": 0}]}
        token_check_response = requests.post(
            self.token_check_endpoint, json=data, headers=self.headers
        )
        return token_check_response.json()["prompts"][0]["contextLength"]
    
    def get_token_count(self, text: str) -> int:
        data = {
            "prompts": [{"model": self.identifier, "prompt": text, "max_tokens": 0}]
        }
        token_check_response = requests.post(
            self.token_check_endpoint, json=data, headers=self.headers
        )
        return token_check_response.json()["prompts"][0]["tokenCount"]

    def __call__(self, model_input, **generation_params):
        """calls the text generation method and returns the generated text."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=model_input,
                **generation_params
            )
            output = response.choices[0].message.content
            output = output.strip()
        except Exception as e:
            print(model_input)
            print(e)
            output = ""
        return output


class OpenAIDecoderModel(APIDecoderModel):
    api_key=os.environ.get("OPENAI_API_KEY")
    base_url="https://api.openai.com/v1"

    def __init__(self, model_id: str, base_url: str) -> None:
        super().__init__(model_id, base_url)
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url
            )
        self.tokenizer = (
            WhitespaceTokenSplitter()
        )        
    def get_tokens(self, text: str) -> list[str]:
        tokens_with_offsets = list(self.tokenizer(text))
        return [token for token, start, end in tokens_with_offsets]

    def get_token_offsets(self, text: str) -> list[tuple[int, int]]:
        tokens_with_offsets = list(self.tokenizer(text))
        return [(start, end) for token, start, end in tokens_with_offsets]

    def _get_model_max_length(self) -> int:
        # info obtained from https://platform.openai.com/docs/models
        max_len_map = {
            "gpt-4o":128000, 
            "gpt-4o-mini":128000, 
            "gpt-4-turbo":128000, 
            "gpt-3.5-turbo":16385,
            }
        return max_len_map[self.model_id]
    
    def get_token_count(self, text: str) -> int:
        # ToDo: add tiktoken implementation
        pass

    def __call__(self, model_input, **generation_params):
        """calls the text generation method and returns the generated text."""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=model_input,
            **generation_params
        )
        output = response.choices[0].message.content
        output = output.strip()
        return output