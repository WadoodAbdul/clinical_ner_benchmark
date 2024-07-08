from .tasks import TaskDataset


class Benchmark:
    def __init__(
        self,
        name: str,
        tasks: list[str],
        clinical_types: list[str],
        description: str | None = None,
        reference: str | None = None,
        citation: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.reference = reference
        self.citation = citation
        self.tasks = tasks  # of TaskDatasets
        self.clinical_types = clinical_types
        self.load_datasets() 

    # def __iter__(self):
    #     return iter(self.tasks)

    # def __len__(self) -> int:
    #     return len(self.tasks)

    # def __getitem__(self, index):
    #     return self.tasks[index]

    def __call__(self, task_identifier):
        return TaskDataset.from_predefined(task_identifier)
    
    def load_datasets(self):
        for task_dataset_name in self.tasks:
            task_dataset = TaskDataset.from_predefined(task_dataset_name)
            setattr(self, task_dataset_name, task_dataset)

    def __getitem__(self, attr):
        return getattr(self, attr)




MEDICS_NER = Benchmark(
    name="MEDICS_NER",
    tasks=[
        "NCBI",
        "CHIA",
        "BIORED",
        "BC5CDR",
        # "BC4CHEMD",
        # "BC2GM",
        # "JNLPBA",
    ],
    clinical_types = ['condition', 'drug', 'procedure', 'measurement', 'gene', 'gene variant'],
    description="",
    reference="",
    citation="""""",
)
