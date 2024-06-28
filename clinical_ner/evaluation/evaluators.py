import json
import os
from collections import defaultdict


import clinical_ner.evaluation.ner_utils as ner_utils
from clinical_ner.tasks import Task


def get_ground_truth_and_predictions(model, dataset, output_dir):
    if dataset.task_type == Task.NAMED_ENTITY_RECOGNITION:
        return ner_utils.get_ground_truth_and_predictions(
            model, dataset.get_evaluation_split(), output_dir
        )


class Evaluator:
    def __init__(
        self, model, benchmark, dataset_wise_config: dict, output_dir: str | None = None
    ) -> None:
        self.model = model
        self.benchmark = benchmark
        self.dataset_wise_config = dataset_wise_config
        self.output_dir = "../data/outputs" if output_dir is None else output_dir            
        self.output_dir = os.path.join(self.output_dir, model.identifier)

        assert set(benchmark.tasks) == set(dataset_wise_config.keys())

    def evaluate_model_on_dataset(self, model, dataset, model_dataset_outputs_dir):
        """
        Returns the evaluation metric of a model on a dataset.
        Note: the model object is expected to be set for the dataset passed.
        """

        ground_truth, predictions, _, _ = get_ground_truth_and_predictions(
            model, dataset, model_dataset_outputs_dir
        )
        evaluation_metrics = dataset.metric.compute_metrics(
            ground_truth,
            predictions,
            labels=list(set(self.model.label_normalization_map.values())),
        )

        dataset.metric.save_metrics(evaluation_metrics, model_dataset_outputs_dir)

        return evaluation_metrics

    def format_and_consolidate_metrics():
        """
        Saves the metrics in the format that will be used by the leaderboard
        """
        pass

    def save_benchmark_metrics(self, dataset_wise_metrics):
        """ 
        Saves the results json file that will be used by the leaderboard.
        The dataset and clinical entity are both macro averages
        """
        benchmark_output_dir = os.path.join(self.output_dir, self.benchmark.name)
        if not os.path.exists(benchmark_output_dir):
            os.makedirs(benchmark_output_dir)

        span_evaluation_criteria = "ent_type"

        clinical_type_results = defaultdict(list)
        dataset_wise_results = defaultdict(list)

        for task in self.benchmark.tasks:
            type_wise_results = dataset_wise_metrics[task]['results_per_tag']
            for entity in self.benchmark.clinical_types:
                entity_result = type_wise_results.get(entity, None)
                if entity_result is None:
                    # model doesn't support that entity
                    entity_f1 = 0
                else:
                    entity_f1 = entity_result[span_evaluation_criteria]["f1"]
                    dataset_wise_results[task].append(entity_f1)
                # model_results.append(entity_f1)
                clinical_type_results[entity].append(entity_f1)
        
        mean = lambda score_list : sum(score_list) / len(score_list)

        benchmark_metrics = {
            "dataset_results" : {k.lower():{'f1':round((mean(v)*100),1)} for k,v in dataset_wise_results.items()},
            "clinical_type_results":{k:{'f1':round((mean(v)*100),1)} for k,v in clinical_type_results.items()},
                             }


        with open(os.path.join(benchmark_output_dir, "results.json"), "w") as f:
            json.dump(benchmark_metrics, f)

    def run(self):
        """
        Runs the evaluation pipeline on the benchmark tasks
        """

        dataset_wise_metrics = {}

        for dataset_name, dataset_config in self.dataset_wise_config.items():
            # sets the attributes of the model object to get outputs aligned to the datasets.
            # This is needed for zeroshot models
            self.model.set_attributes_for_dataset(**dataset_config)

            benchmark_dataset = self.benchmark(dataset_name)

            dataset_outputs_dir = os.path.join(
                self.output_dir, benchmark_dataset.identifier
            )
            if not os.path.exists(dataset_outputs_dir):
                os.makedirs(dataset_outputs_dir)

            evaluation_metrics = self.evaluate_model_on_dataset(
                self.model, benchmark_dataset, dataset_outputs_dir
            )

            dataset_wise_metrics[dataset_name] = evaluation_metrics
            ner_utils.save_metrics(evaluation_metrics, dataset_outputs_dir)

        self.save_benchmark_metrics(dataset_wise_metrics)

        return dataset_wise_metrics