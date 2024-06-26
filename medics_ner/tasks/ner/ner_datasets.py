from abc import ABC, abstractmethod

from datasets import load_dataset

from medics_ner.evaluation.metrics import NERPartialSpanMetric

from ..base_tasks import Task, TaskDataset

TASK_NAME_HF_LINK_MAP = {"NCBI": "m42-health/medics_ner"}


class NERDataset(TaskDataset):
    task_type = Task.NAMED_ENTITY_RECOGNITION
    metric = NERPartialSpanMetric()

    def __init__(self, identifier) -> None:
        super().__init__(identifier)

    @abstractmethod
    def load_data(self):
        pass


class NCBI(NERDataset):
    evaluation_split = "validation"

    def __init__(self, identifier="NCBI") -> None:
        super().__init__(identifier)

    def load_data(self):
        return load_dataset(TASK_NAME_HF_LINK_MAP[self.identifier])

    def get_evaluation_split(self):
        return super().get_evaluation_split()


class CHIA(NERDataset):
    evaluation_split = "validation"

    def __init__(self, identifier="CHIA") -> None:
        super().__init__(identifier)

    def load_data(self):
        return load_dataset(TASK_NAME_HF_LINK_MAP[self.identifier])

    def get_evaluation_split(self):
        return super().get_evaluation_split()
