from abc import ABC, abstractmethod
from enum import Enum


class Task(Enum):
    NAMED_ENTITY_RECOGNITION = "NER"
    NAMED_ENTITY_LINKING = "NEL"
    EVENT_EXTRACTION = "EE"
    RELATION_EXTRACTION = "RE"
    QUESTION_ANSWERING = "QA"
    SUMMARIZATION = "SUM"
    TEXT_CLASSIFICATION = "TXTCLASS"


class TaskDataset(ABC):
    """
    This base class is an abstraction for all datasets(belonging to the task enum above). 
    The class will be used to create a benchmark which will by used by the evaluator class
    """
    def __init__(
        self,
        identifier,
    ) -> None:
        self.identifier = identifier
        self.dataset = self.load_data()

    # @classmethod
    # def from_predefined(cls, identifier, **kwargs):
    #     from .ner_dataset_loader import load_task_dataset

    #     return load_task_dataset(identifier, **kwargs)

    @property
    @abstractmethod
    def metric(self):
        pass

    @property
    @abstractmethod
    def task_type(self):
        pass

    @property
    @abstractmethod
    def evaluation_split(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def get_evaluation_split(self):
        """
        Current implementation is for hf dataset, which is subscriptable.
        Override this method for custom implementation.
        """
        return self.dataset[self.evaluation_split]
