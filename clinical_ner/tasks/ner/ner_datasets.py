from abc import ABC, abstractmethod

from datasets import load_dataset

from clinical_ner.evaluation.metrics import NERPartialSpanMetric
from clinical_ner.tasks.ner import load_and_prepare_dataset
from ..base_tasks import Task, TaskDataset


TASK_NAME_HF_LINK_MAP = {
    "NCBI": "m42-health/m2_ncbi",
    "CHIA": "m42-health/m2_chia",
    "BIORED": "m42-health/m2_biored",
    "BC5CDR": "m42-health/m2_bc5cdr",
    }


class NERDataset(TaskDataset):
    task_type = Task.NAMED_ENTITY_RECOGNITION
    metric = NERPartialSpanMetric()

    def __init__(self, identifier) -> None:
        super().__init__(identifier)

    @abstractmethod
    def load_data(self):
        pass


class NCBI(NERDataset):
    evaluation_split = "test"
    clinical_types = ["condition"]

    def __init__(self, identifier="NCBI") -> None:
        super().__init__(identifier)

    def load_data(self):
        return load_and_prepare_dataset(self.identifier)
        # return load_dataset(TASK_NAME_HF_LINK_MAP[self.identifier])

    def get_evaluation_split(self):
        return super().get_evaluation_split()


class CHIA(NERDataset):
    evaluation_split = "test"
    clinical_types = ["condition", "drug", "procedure", "measurement"]

    def __init__(self, identifier="CHIA") -> None:
        super().__init__(identifier)

    def load_data(self):
        return load_and_prepare_dataset(self.identifier)
        # return load_dataset(TASK_NAME_HF_LINK_MAP[self.identifier])

    def get_evaluation_split(self):
        return super().get_evaluation_split()

class BIORED(NERDataset):
    evaluation_split = "test"
    clinical_types = ["condition", "drug", "gene", "gene variant"]

    def __init__(self, identifier="BIORED") -> None:
        super().__init__(identifier)

    def load_data(self):
        return load_and_prepare_dataset(self.identifier)
        # return load_dataset(TASK_NAME_HF_LINK_MAP[self.identifier])

    def get_evaluation_split(self):
        return super().get_evaluation_split()

        
class BC5CDR(NERDataset):
    evaluation_split = "test"
    clinical_types = ["condition", "drug"]

    def __init__(self, identifier="BC5CDR") -> None:
        super().__init__(identifier)

    def load_data(self):
        return load_and_prepare_dataset(self.identifier)
        # return load_dataset(TASK_NAME_HF_LINK_MAP[self.identifier])

    def get_evaluation_split(self):
        return super().get_evaluation_split()