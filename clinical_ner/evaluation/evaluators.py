import json
import os
from collections import defaultdict
from datetime import datetime

import clinical_ner.evaluation.ner_utils as ner_utils
from clinical_ner.tasks import Task
from clinical_ner.evaluation.metrics import NERObjectType, NEREvaluationMetric


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
        self.execution_time_stamp = str(datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
        self.output_dir = os.path.join(self.output_dir, f"outputs_{self.execution_time_stamp}")

        assert set(benchmark.tasks) == set(dataset_wise_config.keys())

    def evaluate_model_on_dataset(self, model, dataset, model_dataset_outputs_dir:str, metrics_to_compute:list[NEREvaluationMetric]):
        """
        Returns the evaluation metric of a model on a dataset.
        Note: the model object is expected to be set for the dataset passed.
        """

        span_ground_truths, span_predictions, token_ground_truths, token_predictions = get_ground_truth_and_predictions(
            model, dataset, model_dataset_outputs_dir
        )

        metric_wise_result_on_dataset = {}
        for metric in metrics_to_compute:
            if metric.NER_OBJECT_TYPE == NERObjectType.SpanBased:
                ground_truths, predictions = span_ground_truths, span_predictions
            elif metric.NER_OBJECT_TYPE == NERObjectType.TokenBased:
                ground_truths, predictions = token_ground_truths, token_predictions 
            else:
                print(f"{metric.NAME=} does not have a well defined ner object type")

            metric_result = metric.compute_metrics(
                ground_truths, 
                predictions, 
                dataset.clinical_types, 
                model_dataset_outputs_dir
                )
            metric_wise_result_on_dataset[metric.NAME] = metric_result
        # evaluation_metrics = dataset.metric.compute_metrics(
        #     ground_truth,
        #     predictions,
        #     labels=list(set(self.model.label_normalization_map.values())), #should this be the labels in dataset?
        # )

        return metric_wise_result_on_dataset

    def format_and_consolidate_metrics():
        """
        Saves the metrics in the format that will be used by the leaderboard
        """
        pass

    def save_model_inference_config(self, inference_config):
        """
        Saves the config used to run inference. This can be used to reproduce the result.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(os.path.join(self.output_dir, "inference_config.json"), "w") as f:
            json.dump(inference_config, f)
        

    def save_benchmark_metrics(self, dataset_wise_metrics):
        """ 
        Saves the results json file that will be used by the leaderboard.
        The dataset and clinical entity are both macro averages
        """
        evaluation_metric_results = {}

        for dataset_name, dataset_results in dataset_wise_metrics.items():
            for evaluation_metric, eval_metric_results in dataset_results.items():

                evaluation_metric_results[evaluation_metric] = evaluation_metric_results.get(evaluation_metric, {})

                evaluation_metric_results[evaluation_metric]["dataset_results"] = evaluation_metric_results[evaluation_metric].get("dataset_results", {})
                evaluation_metric_results[evaluation_metric]["clinical_type_results"] = evaluation_metric_results[evaluation_metric].get("clinical_type_results", {})

                evaluation_metric_results[evaluation_metric]["dataset_results"][dataset_name.lower()] = {"f1":round(eval_metric_results.score * 100, 2)}
                for clinical_type, clinical_type_results_dict in eval_metric_results['entity_wise_results'].items():
                    clinical_type_score = clinical_type_results_dict['f1']

                    evaluation_metric_results[evaluation_metric]["clinical_type_results"][clinical_type] = evaluation_metric_results[evaluation_metric]["clinical_type_results"].get(clinical_type, [])
                    evaluation_metric_results[evaluation_metric]["clinical_type_results"][clinical_type].append(clinical_type_score)

        mean = lambda score_list : sum(score_list) / len(score_list)
        for evaluation_metric, eval_metric_results in evaluation_metric_results.items():
            eval_metric_results['clinical_type_results'] = {k:{'f1':round((mean(v)*100),2)} for k,v in eval_metric_results['clinical_type_results'].items()}


        with open(os.path.join(self.output_dir, f"{self.benchmark.name.lower()}_results.json"), "w") as f:
            json.dump(evaluation_metric_results, f)

        return evaluation_metric_results

    def run(self):
        """
        Runs the evaluation pipeline on the benchmark tasks
        """

        dataset_wise_metrics = {}
        dataset_wise_configs = {}

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

            evaluation_metric_wise_result_on_dataset = self.evaluate_model_on_dataset(
                self.model, benchmark_dataset, dataset_outputs_dir, self.benchmark.metrics_to_compute
            )

            dataset_wise_metrics[dataset_name] = evaluation_metric_wise_result_on_dataset
            dataset_wise_configs[dataset_name] = self.model.get_inference_config()
            # ner_utils.save_metrics(evaluation_metrics, dataset_outputs_dir)

        self.save_model_inference_config(dataset_wise_configs)
        print(dataset_wise_metrics)
        benchmark_metrics = self.save_benchmark_metrics(dataset_wise_metrics)


        return dataset_wise_metrics, benchmark_metrics