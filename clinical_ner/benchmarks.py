from .evaluation import (
    SPAN_AND_TOKEN_METRICS_FOR_NER,
    NEREvaluationMetric,
)
from .tasks import (
    BC5CDR,
    BC5CDR_PROMPT_ENGINEERING,
    BIORED,
    BIORED_PROMPT_ENGINEERING,
    CHIA,
    CHIA_PROMPT_ENGINEERING,
    NCBI,
    NCBI_PROMPT_ENGINEERING,
)


class Benchmark:
    def __init__(
        self,
        name: str,
        tasks: list[str],
        clinical_types: list[str],
        metrics_to_compute: list[NEREvaluationMetric],
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
        self.metrics_to_compute = metrics_to_compute
        self.load_datasets()

    # def __iter__(self):
    #     return iter(self.tasks)

    # def __len__(self) -> int:
    #     return len(self.tasks)

    # def __getitem__(self, index):
    #     return self.tasks[index]

    def __call__(self, task_identifier):
        return getattr(self, task_identifier)

    def load_datasets(self):
        tasks = []
        for task in self.tasks:
            task = task()
            task_dataset_name = task.identifier
            tasks.append(task_dataset_name)
            setattr(self, task_dataset_name, task)

        self.tasks = tasks

    def __getitem__(self, attr):
        return getattr(self, attr)


NCER = Benchmark(
    name="NCER",
    tasks=[
        NCBI,
        CHIA,
        BIORED,
        BC5CDR,
    ],
    clinical_types=[
        "condition",
        "drug",
        "procedure",
        "measurement",
        "gene",
        "gene variant",
    ],
    metrics_to_compute=SPAN_AND_TOKEN_METRICS_FOR_NER,
    description="",
    reference="",
    citation="""""",
)

PROMPT_TESTER = Benchmark(
    name="PROMPT_TESTER",
    tasks=[
        NCBI_PROMPT_ENGINEERING,
        CHIA_PROMPT_ENGINEERING,
        BIORED_PROMPT_ENGINEERING,
        BC5CDR_PROMPT_ENGINEERING,
    ],
    clinical_types=[
        "condition",
        "drug",
        "procedure",
        "measurement",
        "gene",
        "gene variant",
    ],
    metrics_to_compute=SPAN_AND_TOKEN_METRICS_FOR_NER,
    description="",
    reference="",
    citation="""""",
)
