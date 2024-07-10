import os
from typing import Literal 
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI


dotenv_path = os.path.join(os.path.abspath('../../'), '.env')
load_dotenv(dotenv_path)

class LLMClient():
    def __init__(self, model_name:Literal["Meta-Llama-3-70b-Instruct", "Mixtral-8x7B-Instruct-v0.1", "Llama3-Med42-DPO-70B", "gpt-4o", "models/gemini-1.5-pro"]="Meta-Llama-3-70b-Instruct", temperature:float=.0, max_output_tokens:int=6_000) -> None:
        self.model_id = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens 
        self._set_client()


    def _set_client(self) -> None:
        if self.model_id == "models/gemini-1.5-pro":
            self.client = self._set_gemini_client()
        elif self.model_id == "gpt-4o":
            self.client = self._set_openai_client()
        else:
            self.client = self._set_local_client()
    

    def get_score(self, prompt:str, system_prompt:str="You are a helpful assistant.") -> str:
        if self.model_id == "models/gemini-1.5-pro":
            response = self.client.generate_content(prompt)
            return response.text
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            # response_format={ "type": "text" },
            # response_format={ "type": "json_object" },
            messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": prompt }
            ]
        )
        response = response.choices[0].message.content
        response = response.strip()
        return response


    def _set_gemini_client(self) -> genai.GenerativeModel:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                # "response_mime_type": "application/json" #"text/plain",
            },
        )
        return client


    def _set_openai_client(self) -> OpenAI:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return client
    

    def _set_local_client(self) -> OpenAI:
        client = OpenAI(api_key="EMPTY", base_url="http://dev-openai-api.med42.ai/v1")
        return client


    def get_model_id(self):
        return self.model_id
