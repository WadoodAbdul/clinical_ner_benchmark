from dataclasses import dataclass

@dataclass
class NERDatasetResult:
    """
    The evaluation result output of a dataset 
    example = {
        "f1":99, 
        "entity_type_results" : {
            "entity_1" : {"f1":43}, 
            "entity_2":{"f1":56}}
            }
        }
    """
    score: float
    entity_wise_results: dict[str, dict[str, float]]

    def __getitem__(self, item):
        return getattr(self, item)


