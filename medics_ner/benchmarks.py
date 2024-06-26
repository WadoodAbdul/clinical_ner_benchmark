from tasks import TaskDataset


class Benchmark:
    def __init__(
        self,
        name: str,
        tasks: list[str],
        description: str | None = None,
        reference: str | None = None,
        citation: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.reference = reference
        self.citation = citation
        self.tasks = tasks  # of TaskDatasets

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index):
        return self.tasks[index]

    def __call__(self, task_identifier):
        return TaskDataset.from_predefined(task_identifier)


MEDICS_NER = Benchmark(
    name="MEDICS_NER",
    tasks=[
        "NCBI",
        # "CHIA",
        # "BIORED",
        # "BC5CDR",
        # "BC4CHEMD",
        # "BC2GM",
        # "JNLPBA",
    ],
    description="",
    reference="",
    citation="""""",
)
