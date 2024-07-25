import os
from abc import ABC, abstractmethod
from typing import Literal
from jinja2 import Environment, FileSystemLoader


from clinical_ner.models.output_parsers import (
    parse_from_list_of_span_texts,
    parse_from_html_spans,
                                                
)


class PromptTemplate(ABC):
    """ Abstract base class defining the functionality of the Prompt class"""
    prompt_template_filename:str
    # parsing_function:Optional[Callable] this gets registered as a self method and creates num of args issues later
    decoder_model_type:Literal["base", "chat"]
    loop_for_each_entity:bool

    def __init__(self) -> None:
        super().__init__()
        self.prompt_template_filename = self.prompt_template_filename
        # self.parsing_function = self.parsing_function
        self.decoder_model_type = self.decoder_model_type
        self.loop_for_each_entity = self.loop_for_each_entity

        self.prompt_template = self.load_prompt_template(self.prompt_template_filename)

    @classmethod
    def from_predefined(cls, prompt_template_identifier):
        "Returns an instance of the prompt template class"
        match prompt_template_identifier:
            case "universal_ner":
                return UniversalNERPromptTemplate()
            case "llm_html_highlighted_spans":
                return LLMHTMLSpansPromptTemplate()
            case "llama_70B_ner":
                return LlamaNerPromptTemplate()
            case "llm_html_highlighted_spans_v1":
                return LLMHTMLSpansPromptTemplateV1()
            case _:
                raise NotImplementedError(
                    f"prompt_template not implemented for {prompt_template_identifier=}"
                    )
        pass
    
    @staticmethod
    @abstractmethod
    def parsing_function(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_prompt_template(prompt_template_filename: str) -> str:
        """Returns the prompt with kwargs substititued in the string"""
        env = Environment(
            loader = FileSystemLoader(
                os.path.abspath(
                    '../data/inputs/prompt_templates'
                    )
                )
            )

        template = env.get_template(prompt_template_filename+'.jinja')
        return template

    # def load_prompt_with_variables(self, prompt_template, **kwargs) -> str:
    #     return prompt_template.render(kwargs)
    
    @abstractmethod
    def get_expected_sequence_length_text(self, text) -> int:
        """
        Returns the expected text tokens in the prepared_input + expected_output.
        This will be used to pass to the tokenizer, get expected tokens num, which will be used for chunking
        Different prompts can warrant a different generation length to solve the task. 
        """
        pass

    @abstractmethod
    def create_model_input(self, *args, **kwargs):
        """Returns the model input that will be used for inference. 
        This is either a string or a list of dictionaries based on the decoder model type the prompt caters to.
        """
        pass


class UniversalNERPromptTemplate(PromptTemplate):
    prompt_template_filename="universal_ner"
    # parsing_function=parse_from_list_of_span_texts
    decoder_model_type="base"
    loop_for_each_entity=True
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def parsing_function(*args, **kwargs):
        return parse_from_list_of_span_texts(*args, **kwargs)

    def get_expected_sequence_length_text(self, text) -> str:
        return self.create_model_input(text=text, entity="dummy_entity") + " word"*300 # 300 is arbitrary here, this should be proportional to the number of words in the entities in the text

    def create_model_input(self, **prompt_variables) -> str:
        return self.prompt_template.render({
            "text":prompt_variables['text'],
            "entity":prompt_variables['entity'],
            })


class LLMHTMLSpansPromptTemplate(PromptTemplate):
    prompt_template_filename="llm_html_highlighted_spans"
    # parsing_function=parse_from_html_spans
    decoder_model_type="chat"
    loop_for_each_entity=False
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def parsing_function(*args, **kwargs):
        return parse_from_html_spans(*args, **kwargs)

    def get_expected_sequence_length_text(self, text) -> str:
        model_input = self.create_model_input(text=text)
        input_text = " ".join([ins["content"] for ins in model_input])

        return input_text + text + '<span class="disease" > </span >'*300
    
    def create_model_input(self, **prompt_variables) -> list:
        system_instruction = self.prompt_template.render(
            {
                "is_system_instruction": True
            }
        )
        user_instruction = self.prompt_template.render(
            {
                "is_user_instruction": True,
                "text": prompt_variables["text"],
            }
        )

        messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_instruction},
            ]

        return messages


class LlamaNerPromptTemplate(PromptTemplate):
    prompt_template_filename="llama_70B_ner"
    # parsing_function=parse_from_list_of_span_texts
    decoder_model_type="chat"
    loop_for_each_entity=True
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def parsing_function(*args, **kwargs):
        return parse_from_list_of_span_texts(*args, **kwargs)
    
    def get_expected_sequence_length_text(self, text) -> str:
        model_input = self.create_model_input(text=text, entity="disease")
        input_text = " ".join([ins["content"] for ins in model_input])

        return input_text +  " word"*300
    
    def create_model_input(self, **prompt_variables) -> list:
        system_instruction = self.prompt_template.render(
            {
                "is_system_instruction": True
            }
        )
        user_instruction = self.prompt_template.render(
            {
                "is_user_instruction": True,
                "text": prompt_variables["text"],
                "entity": prompt_variables["entity"]
            }
        )

        messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_instruction},
            ]

        return messages

class LLMHTMLSpansPromptTemplateV1(PromptTemplate):
    prompt_template_filename="llm_html_highlighted_spans_v1"
    decoder_model_type="chat"
    loop_for_each_entity=False
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def parsing_function(*args, **kwargs):
        return parse_from_html_spans(*args, **kwargs)

    def get_expected_sequence_length_text(self, text) -> str:
        model_input = self.create_model_input(text=text)
        input_text = " ".join([ins["content"] for ins in model_input])

        return input_text + text + '<span class="disease" > </span >'*300
    
    def create_model_input(self, **prompt_variables) -> list:
        system_instruction = self.prompt_template.render(
            {
                "is_system_instruction": True
            }
        )
        user_instruction = self.prompt_template.render(
            {
                "is_user_instruction": True,
                "text": prompt_variables["text"],
            }
        )

        messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_instruction},
            ]

        return messages
