import json
import os
from datetime import datetime

import pandas as pd
from datasets import Dataset, load_from_disk
from nervaluate import Evaluator

# from sklearn.metrics import classification_report
# from seqeval.metrics import classification_report
from sklearn.metrics import (
    classification_report,  # sklearn and torch are not on talking
)
from tqdm import tqdm

EVALUATION_RESULTS_FOLDER_PATH = "../data/evaluation_results/"


def save_dict_to_json(dict_variable, file_path):
    with open(file_path, "w") as fp:
        json.dump(dict_variable, fp)


def calculate_f1_score(p, r):
    return (2 * p * r) / (p + r)


def get_span_ground_truth_and_predictions(
    ner_processor, eval_dataset, model_identifier, dataset_identifier, run_identifier
):
    predictions = []
    ground_truths = []

    for data_point in tqdm(eval_dataset):
        dp_pred = ner_processor(data_point["text"])  # , threshold=0.3)

        predictions.append([span.__dict__ for span in dp_pred.spans])
        ground_truths.append(
            data_point["entities"]
        )  # this would always be entites for standardized dataset

    eval_dataset = eval_dataset.add_column("ground_truth_tokens", ground_truths)
    eval_dataset = eval_dataset.add_column("predicted_tokens", predictions)

    eval_dataset.save_to_disk(
        f"../data/evaluation_results/{model_identifier}/{dataset_identifier}/span_level/{run_identifier}/outputs/"
    )
    return ground_truths, predictions


def get_token_ground_truth_and_predictions(
    ner_processor, eval_dataset, model_identifier, dataset_identifier, run_identifier
):
    predictions = []
    ground_truths = []

    for data_point in tqdm(eval_dataset):
        datapoint_pred = ner_processor(data_point["text"])

        token_labels = ner_processor.get_token_level_ner_output_from_spans(
            datapoint_pred
        )
        predictions.append([label for _, label in token_labels])

        token_labels = ner_processor.get_token_level_ner_output_from_spans(
            ner_spans=data_point["entities"], parent_text=data_point["text"]
        )

        ground_truths.append(
            [label for _, label in token_labels]
        )  # this would always be entites for standardized dataset
    eval_dataset = eval_dataset.add_column("ground_truth_tokens", ground_truths)
    eval_dataset = eval_dataset.add_column("predicted_tokens", predictions)

    eval_dataset.save_to_disk(
        f"../data/evaluation_results/{model_identifier}/{dataset_identifier}/token_level/{run_identifier}/outputs/"
    )

    return ground_truths, predictions


def load_ground_truth_and_predictions(saved_dataset_path):
    dataset = load_from_disk(saved_dataset_path)

    return (
        dataset["span_ground_truths"],
        dataset["span_predictions"],
        dataset["token_ground_truths"],
        dataset["token_predictions"],
    )


def get_ground_truth_and_predictions(
    ner_processor, eval_dataset, model_dataset_folder_path
):
    span_predictions = []
    span_ground_truths = []

    token_predictions = []
    token_ground_truths = []

    for data_point in tqdm(eval_dataset):
        datapoint_pred = ner_processor(data_point["text"])

        pred_token_labels = ner_processor.get_token_level_ner_output_from_spans(
            datapoint_pred
        )
        span_predictions.append([span.__dict__ for span in datapoint_pred.spans])
        token_predictions.append([label for _, label in pred_token_labels])

        gt_token_labels = ner_processor.get_token_level_ner_output_from_spans(
            ner_spans=data_point["entities"], parent_text=data_point["text"]
        )
        span_ground_truths.append(data_point["entities"])
        token_ground_truths.append(
            [label for _, label in gt_token_labels]
        )  # this would always be entites for standardized dataset
    eval_dataset = eval_dataset.add_column("ground_truth_spans", span_ground_truths)
    eval_dataset = eval_dataset.add_column("predicted_spans", span_predictions)
    eval_dataset = eval_dataset.add_column("ground_truth_tokens", token_ground_truths)
    eval_dataset = eval_dataset.add_column("predicted_tokens", token_predictions)

    save_path = os.path.join(model_dataset_folder_path, "outputs")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    eval_dataset.save_to_disk(save_path)

    return span_ground_truths, span_predictions, token_ground_truths, token_predictions


def explode_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def get_span_level_metrics(span_ground_truths, span_predictions, labels):
    evaluator = Evaluator(span_ground_truths, span_predictions, tags=labels)
    results, results_per_tag = evaluator.evaluate()
    print(results)
    print(results_per_tag)
    print(f'f1-score: {results["ent_type"]["f1"]}')

    return {
        "results": results,
        "results_per_tag": results_per_tag,
    }


def get_token_level_metrics(token_gt, token_preds, labels):
    gt, preds = explode_list(token_gt), explode_list(token_preds)

    print(classification_report(gt, preds))
    print(pd.crosstab(gt, preds))
    # print(f"{labels=} ")
    report = classification_report(gt, preds, labels=labels, output_dict=True)
    if "accuracy" in report:
        report.update(
            {
                "accuracy": {
                    "precision": None,
                    "recall": None,
                    "f1-score": report["accuracy"],
                    "support": report["macro avg"]["support"],
                }
            }
        )
    cls_report = pd.DataFrame(report).transpose()
    confusion_matrix = pd.crosstab(gt, preds)
    f1_score = cls_report.loc["micro avg"]["f1-score"]
    print(f"{f1_score=}")

    return {
        "cls_report": cls_report,
        "confusion_matrix": confusion_matrix,
    }


def get_metrics(
    span_ground_truths: list = None,
    span_predictions: list = None,
    token_ground_truths: list = None,
    token_predictions: list = None,
    labels: list = None,
):
    """takes in 4 lists and return the whole results(span and token level)"""

    assert labels is not None, "Labels list for evalution cannot be empty"

    metrics = {}
    # Span level
    if span_predictions is not None:
        metrics["span_metrics"] = get_span_level_metrics(
            span_ground_truths, span_predictions, labels
        )

    # Token level
    if token_predictions is not None:
        metrics["token_metrics"] = get_token_level_metrics(
            token_ground_truths, token_predictions, labels
        )
    return metrics


def save_metrics(span_metrics, model_dataset_folder_path):
    # if (token_metrics := evaluation_metrics.get("token_metrics", None)) is not None:
    #     save_path = os.path.join(model_dataset_folder_path, "token_level")
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     token_metrics["cls_report"].round(4).to_csv(
    #         os.path.join(save_path, "classification_report.tsv"),
    #         sep="\t",
    #     )
    #     token_metrics["confusion_matrix"].to_csv(
    #         os.path.join(save_path, "confusion_matrix.csv"),
    #     )

    # if (span_metrics := evaluation_metrics.get("span_metrics", None)) is not None:
    save_path = os.path.join(model_dataset_folder_path, "span_level/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_dict_to_json(
        dict_variable=span_metrics["results"],
        file_path=os.path.join(save_path, "results.json"),
    )
    save_dict_to_json(
        dict_variable=span_metrics["results_per_tag"],
        file_path=os.path.join(
            save_path,
            "results_per_tag.json",
        ),
    )


def evaluate(
    ner_processor, eval_dataset, model_dataset_folder_path, run_inference=True
):
    """Returns span and token level evaluation metrics.
    If run_inference is set to True, will use the ner_processor to run model inference else will try loading from model_dataset_folder_path
    """
    if not os.path.exists(model_dataset_folder_path):
        if run_inference:
            print(f"Run Exists and {run_inference=}, outputs will be overwritten")
        os.makedirs(model_dataset_folder_path)

    if run_inference:
        (
            span_ground_truths,
            span_predictions,
            token_ground_truths,
            token_predictions,
        ) = get_ground_truth_and_predictions(
            ner_processor, eval_dataset, model_dataset_folder_path
        )
    else:
        saved_dataset_path = os.path.join(model_dataset_folder_path, "outputs/")
        (
            span_ground_truths,
            span_predictions,
            token_ground_truths,
            token_predictions,
        ) = load_ground_truth_and_predictions(saved_dataset_path)

    evaluation_results = get_metrics(
        span_ground_truths,
        span_predictions,
        token_ground_truths,
        token_predictions,
        labels=list(set(ner_processor.label_normalization_map.values())),
    )

    save_metrics(evaluation_results, model_dataset_folder_path)

    return evaluation_results
