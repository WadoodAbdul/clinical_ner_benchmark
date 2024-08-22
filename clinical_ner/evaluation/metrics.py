import os
from abc import ABC, abstractmethod
from enum import Enum

from nervaluate import Evaluator
from sklearn.metrics import (
    classification_report,  # sklearn and torch are not on talking
)
from .utils import save_dict_to_json, calculate_f1_score, explode_list
from .results_dataclasses import NERDatasetResult


class NERObjectType(Enum):
    SpanBased = "SpanBased"
    TokenBased = "TokenBased"


class NEREvaluationMetric(ABC):
    """
    should have a name/identifier?
    should have a compute metric method
    shuould have explanation string
    """

    @abstractmethod
    def compute_metrics(self) -> NERDatasetResult:
        """
        This should return {"f1":f1_score, "entity_type_results" : {"entity_1" : {"f1":float}, "entity_2":{"f1":float}}}
        """
        pass

    @abstractmethod
    def save_metrics(self) -> None:
        pass


class SpanBasedMetric(NEREvaluationMetric):
    NAME = ""
    SUB_TYPE = ""
    INFO= ""
    NER_OBJECT_TYPE = NERObjectType.SpanBased

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def compute_metrics(cls, span_ground_truths, span_predictions, labels, save_dir:str=None) -> NERDatasetResult:
        evaluator = Evaluator(span_ground_truths, span_predictions, tags=labels)
        results, results_per_tag = evaluator.evaluate()
        # print(results)
        # print(results_per_tag)

        if save_dir is not None:
            cls.save_metrics(
                metric_results = {
                    "results" : results,
                    "results_per_tag" : results_per_tag
                },
                model_dataset_folder_path=save_dir
            )

        dataset_score = results[cls.SUB_TYPE]["f1"]
        print(f'{cls.NAME} f1-score: {dataset_score}')

        entity_wise_results = {}
        for entity_type, entity_type_result in results_per_tag.items():
            entity_wise_results[entity_type] = {"f1" : entity_type_result[cls.SUB_TYPE]["f1"]}

        return NERDatasetResult(
            score=dataset_score,
            entity_wise_results=entity_wise_results
        )
    
    @classmethod    
    def save_metrics(cls, metric_results:dict, model_dataset_folder_path:str):

        save_path = os.path.join(model_dataset_folder_path, cls.NAME)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_dict_to_json(
            dict_variable=metric_results["results"],
            file_path=os.path.join(save_path, "results.json"),
        )
        save_dict_to_json(
            dict_variable=metric_results["results_per_tag"],
            file_path=os.path.join(
                save_path,
                "results_per_tag.json",
            ),
        )


class TokenBasedMetric(NEREvaluationMetric):
    NAME = ""
    SUB_TYPE = ""
    INFO= ""
    NER_OBJECT_TYPE = NERObjectType.TokenBased

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def compute_metrics(cls, token_ground_truths, token_predictions, labels:list, save_dir:str=None) -> NERDatasetResult:
        gt, preds = explode_list(token_ground_truths), explode_list(token_predictions)

        report = classification_report(gt, preds, labels=labels, output_dict=True)

        if save_dir is not None:
            cls.save_metrics(
                metric_results=report,
                model_dataset_folder_path=save_dir
            )

        f1_score = report[cls.SUB_TYPE]['f1-score']
        print(f"{cls.NAME} f1_score: {f1_score}")

        entity_wise_results = {}
        for label in labels:
            entity_wise_results[label] = {
                "f1" : report[label]["f1-score"]
                }

        return NERDatasetResult(
            score=f1_score,
            entity_wise_results=entity_wise_results
        )
    
    @classmethod
    def save_metrics(cls, metric_results:dict, model_dataset_folder_path:str) -> None:
        save_path = os.path.join(model_dataset_folder_path, cls.NAME)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_dict_to_json(
            dict_variable=metric_results,
            file_path=os.path.join(save_path, "results.json"),
        )

class SpanBasedWithPartialOverlapMetric(SpanBasedMetric):
    NAME = "SpanBasedWithPartialOverlap"
    SUB_TYPE = "ent_type" # this is an artifact of the library used for evaluation
    INFO = ""


class SpanBasedWithExactOverlapMetric(SpanBasedMetric):
    NAME = "SpanBasedWithExactOverlap"
    SUB_TYPE = "strict" # this is an artifact of the library used for evaluation
    INFO = ""


class TokenBasedWithMacroAverageMetric(TokenBasedMetric):
    NAME = "TokenBasedWithMacroAverage"
    SUB_TYPE = "macro avg"
    INFO = ""


class TokenBasedWithMicroAverageMetric(TokenBasedMetric):
    NAME = "TokenBasedWithMicroAverage"
    SUB_TYPE = "micro avg"
    INFO = ""


class TokenBasedWithWeightedAverageMetric(TokenBasedMetric):
    NAME = "TokenBasedWithWeightedAverage"
    SUB_TYPE = "weighted avg"
    INFO = ""

SPAN_AND_TOKEN_METRICS_FOR_NER = [
        SpanBasedWithPartialOverlapMetric, 
        SpanBasedWithExactOverlapMetric,
        TokenBasedWithMacroAverageMetric,
        TokenBasedWithMicroAverageMetric,
        TokenBasedWithWeightedAverageMetric,
        ]