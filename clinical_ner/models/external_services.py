import logging
import os
import json

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

from .extractor_base import GenericSpanExtractor
from .mixins import WhiteSpaceTokenizerMixin
from .model_output_dataclass import (
    IntermediateOutputs,
    NERChunkOutput,
    NEROutput,
)
from .span_dataclasses import NERSpan, NERSpans

dotenv_path = os.path.join(os.path.abspath("../"), ".env")
print(dotenv_path)
load_dotenv(dotenv_path)

logging.getLogger("azure.core").setLevel(logging.ERROR)


def convert_entity_result_to_dict(azure_entity_result):
    azure_entity_result = azure_entity_result.__dict__

    if not azure_entity_result["data_sources"]:
        azure_entity_result["data_sources"] = []

    if azure_entity_result["assertion"] and not isinstance(azure_entity_result["assertion"], dict):
        azure_entity_result["assertion"] = azure_entity_result["assertion"].__dict__

    azure_entity_result["data_sources"] = [
        ds.__dict__ if not isinstance(ds, dict) else ds
        for ds in azure_entity_result["data_sources"]
    ]
    return azure_entity_result


def convert_relation_role_to_dict(relation_role):
    relation_role_dict = relation_role.__dict__
    # print(relation_role_dict.keys())
    relation_role_dict["entity"] = convert_entity_result_to_dict(
        relation_role_dict["entity"]
    )
    return relation_role_dict


def convert_relation_result_to_dict(relation_result):
    relation_result_dict = relation_result.__dict__

    relation_result_dict["roles"] = [
        convert_relation_role_to_dict(role) for role in relation_result_dict["roles"]
    ]
    return relation_result_dict


def convert_azure_result_to_dict(azure_result):
    azure_result_dict = azure_result.__dict__
    azure_result_dict["entities"] = [
        convert_entity_result_to_dict(entity_result)
        for entity_result in azure_result_dict["entities"]
    ]
    azure_result_dict["entity_relations"] = [
        convert_relation_result_to_dict(relation_result)
        for relation_result in azure_result_dict["entity_relations"]
    ]

    if azure_result_dict['warnings']:
        azure_result_dict['warnings'] = [
            warn.__dict__ if not isinstance(warn, dict) else warn
            for warn in azure_result_dict['warnings']
            ]

    return azure_result_dict


class AzureTextAnalyticsHealthcareService(
    GenericSpanExtractor, WhiteSpaceTokenizerMixin
):
    def __init__(self, label_normalization_map=None) -> None:
        super().__init__()
        self.identifier = "azure_text_analytics"
        self.key = os.environ.get("AZURE_KEY")
        self.endpoint = os.environ.get("AZURE_ENDPOINT")
        self.client = self.get_text_analytics_client()
        self.label_normalization_map = label_normalization_map

    def get_text_analytics_client(self):
        ta_credential = AzureKeyCredential(self.key)
        text_analytics_client = TextAnalyticsClient(
            endpoint=self.endpoint, credential=ta_credential
        )
        return text_analytics_client
    
    def get_inference_config(self) -> dict:
        import azure.ai.textanalytics as lib

        return {
            "label_normalization_map": self.label_normalization_map,
            "library_version": lib.__version__,
        }

    def standardize_azure_ner_output(self, azure_result_dict, parent_text) -> NEROutput:
        spans = []
        for entity_dict in azure_result_dict["entities"]:
            spans.append(
                NERSpan(
                    start=entity_dict["offset"],
                    end=entity_dict["offset"] + entity_dict["length"],
                    span_text=entity_dict["text"],
                    label=entity_dict["category"],
                    confidence=entity_dict["confidence_score"],
                )
            )
        return NERSpans(parent_text=parent_text, spans=spans)

    def extract_spans_from_chunk(self, text: str, **kwargs) -> NERChunkOutput:
        poller = self.client.begin_analyze_healthcare_entities([text])
        result = poller.result()

        docs = [doc for doc in result if not doc.is_error]

        azure_ner_output = docs[0]
        azure_result_dict = convert_azure_result_to_dict(azure_ner_output)
        ner_spans = self.standardize_azure_ner_output(azure_result_dict, text)

        # check if azure_result_dict can be serialized , if not 
        try:
            _ = json.dumps(azure_result_dict)
        except Exception as e:
            print(azure_result_dict['warnings'])
            azure_result_dict = {k:v for k,v in azure_result_dict.items() if k in ('entities', 'entity_relations')}
            print(e)


        if self.label_normalization_map:
            normalized_spans = []
            for ner_span in ner_spans.spans:
                if not self.label_normalization_map.get(ner_span.label, None):
                    continue

                normalized_spans.append(
                    NERSpan(
                        start=ner_span.start,
                        end=ner_span.end,
                        span_text=ner_span.span_text,
                        label=self.label_normalization_map[ner_span.label],
                        confidence=-1,
                    )
                )
            ner_spans = NERSpans(parent_text=text, spans=normalized_spans)

        return NERChunkOutput(
            ner_spans=ner_spans,
            intermediate_outputs=IntermediateOutputs(
                model_io=[azure_result_dict],
                entities=[],
                label_normalization_map=self.label_normalization_map,
            ),
        )
