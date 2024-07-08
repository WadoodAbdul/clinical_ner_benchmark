import os
from abc import ABC, abstractmethod

from nervaluate import Evaluator

from .utils import save_dict_to_json, calculate_f1_score, explode_list


class BaseMetric(ABC):
    """
    should have a name/identifier?
    should have a compute metric method
    shuould have explanation string
    """

    @abstractmethod
    def compute_metrics(self):
        pass


class NERPartialSpanMetric(BaseMetric):
    INFO = ""
    SPAN_METRIC_TYPE = "ent_type"

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def compute_metrics(span_ground_truths, span_predictions, labels):
        evaluator = Evaluator(span_ground_truths, span_predictions, tags=labels)
        results, results_per_tag = evaluator.evaluate()
        print(results)
        print(results_per_tag)
        print(f'f1-score: {results[NERPartialSpanMetric.SPAN_METRIC_TYPE]["f1"]}')

        return {
            "results": results,
            "results_per_tag": results_per_tag,
        }
    
    @staticmethod
    def save_metrics(span_metrics, model_dataset_folder_path):

        # if (span_metrics := evaluation_metrics.get("span_metrics", None)) is not None:
        save_path = os.path.join(model_dataset_folder_path, "partial_span_level_metrics/")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_dict_to_json(
            dict_variable=span_metrics["results"],
            file_path=os.path.join(save_path, "results.json"),
        )
        save_dict_to_json(
            dict_variable=span_metrics["results_per_tag"],
            file_path=os.path.join(
                save_path,
                "results_per_tag.json",
            ),
        )
