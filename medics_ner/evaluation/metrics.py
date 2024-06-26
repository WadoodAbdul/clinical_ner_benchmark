from abc import ABC, abstractmethod

from nervaluate import Evaluator


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

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def compute_metrics(span_ground_truths, span_predictions, labels):
        evaluator = Evaluator(span_ground_truths, span_predictions, tags=labels)
        results, results_per_tag = evaluator.evaluate()
        print(results)
        print(results_per_tag)
        print(f'f1-score: {results["partial"]["f1"]}')

        return {
            "results": results,
            "results_per_tag": results_per_tag,
        }
