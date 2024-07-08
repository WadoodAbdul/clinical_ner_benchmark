import re
from functools import partial
from typing import Callable, List

from datasets import Dataset, load_dataset


SPLITS = [
    "train",
    "test",
    "validation",
]

DATASET_INFO_MAP = {
        "bc2gm": {
            "hf_identifier": "spyysalo/bc2gm_corpus",
            "split_needed": False,
            "token_mapping_needed": True,
            "label_map": {"gene": "gene"},
        },
        "bc4chemd": {
            "hf_identifier": "chintagunta85/bc4chemd",
            "split_needed": False,
            "token_mapping_needed": True,
            "label_map": {"Chemical": "drug"},
        },
        "jnlpba": {
            "hf_identifier": "jnlpba",
            "split_needed": True,
            "token_mapping_needed": True,
        },
        "chia": {
            "hf_identifier": "bigbio/chia",
            "split_needed": True,
            "token_mapping_needed": False,
            "label_map": {
                # "Device": "device",
                "Condition": "condition",
                # 'Mood':'',
                # 'Temporal':'',
                # 'Negation':'',
                # 'Observation':'',
                # "Qualifier": "",
                "Drug": "drug",
                # 'Scope':'',
                "Procedure": "procedure",
                # 'Reference_point':'',
                # 'Person':'',
                # 'Value':'',
                # 'Multiplier':'',
                "Measurement": "measurement",
                # 'Visit':'',
            },
        },
        "biored": {
            "hf_identifier": "bigbio/biored",
            "split_needed": False,
            "token_mapping_needed": False,
            "label_map": {
                "GeneOrGeneProduct": "gene",
                "DiseaseOrPhenotypicFeature": "condition",
                # "OrganismTaxon": "",
                "ChemicalEntity": "drug",
                "SequenceVariant": "gene variant",
                # "CellLine": "",
            },
        },
        "bc5cdr": {
            "hf_identifier": "bigbio/bc5cdr",
            "split_needed": False,
            "token_mapping_needed": False,
            "label_map": {
                "Chemical": "drug",
                "Disease": "condition",
            },
        },
        "ncbi": {
            "hf_identifier": "bigbio/ncbi_disease",
            "split_needed": False,
            "token_mapping_needed": False,
            "label_map": {
                "CompositeMention": "condition",
                "DiseaseClass": "condition",
                "Modifier": "condition",
                "SpecificDisease": "condition",
            },
        },
    }

SPANS_DROPPED = 0
MANUAL_LABEL_MAP = {
    "bc2gm": {0: "O", 1: "gene", 2: "gene"},
}


def clean_text(text):
    "cleans training whitespaces and na from string"
    return text.strip().lower().replace("na", "")


def get_all_matches(span_text, document_text, ignore_case=False):
    """returns a list of span matches in the form of [(start, end)]"""
    if ignore_case:
        flag = re.IGNORECASE.value
    else:
        flag = 0

    re_matches = list(re.finditer(re.escape(span_text), document_text, flag))
    return [match.span() for match in re_matches]


def update_example_from_tokens_and_labels(example, int_label_name_map):
    text = ""
    ner_entites = []
    span_text = ""
    span_started = False
    start_idx = None
    span_ids = []
    label_str = ""
    for i, (token, label) in enumerate(zip(example["tokens"], example["ner_tags"])):
        if label != 0:  # token belonging to span
            if label_str != "" and label_str != int_label_name_map[label]:
                # meaning we have consecutive ner spans of diff lables(non null),
                # we should store prev span and reset vars
                end_idx = start_idx + len(span_text.strip())
                span_ids = set(span_ids)
                span_labels = set([int_label_name_map[i] for i in span_ids])
                # if len(span_labels) != 1:
                    # print(
                    #     f"ERROR - multiple labels for a single span, {span_labels=}, {span_text=}, {span_ids=}"
                    # )
                span_label = list(span_labels)[0]
                ner_entites.append(
                    {
                        "text": span_text.strip(),
                        "start": start_idx,
                        "end": end_idx,
                        "label": span_label,
                    }
                )
                span_started = False
                span_ids = []
                label_str = ""
                span_text = ""

            if not span_started:  # first token of span
                start_idx = max(0, len(text))  # len of text till now
                label_str = int_label_name_map[label]

            span_text += token + " "
            span_started = True
            span_ids.append(label)

        else:  # non span token
            if (
                span_started
            ):  # the previous token belonged to a span, so span ended in the prev token
                end_idx = start_idx + len(span_text.strip())
                span_ids = set(span_ids)
                span_labels = set([int_label_name_map[i] for i in span_ids])
                # if len(span_labels) != 1:
                    # print(
                    #     f"ERROR - multiple labels for a single span, {span_labels=}, {span_text=}, {span_ids=}"
                    # )
                span_label = list(span_labels)[0]
                ner_entites.append(
                    {
                        "text": span_text.strip(),
                        "start": start_idx,
                        "end": end_idx,
                        "label": span_label,
                    }
                )
                span_started = False
                span_ids = []
                label_str = ""

            span_text = ""

        text = text + token + " "

    text = text.strip()
    example["entities"] = ner_entites
    example["document_id"] = example["id"]
    example["text"] = text

    del example["tokens"]
    del example["ner_tags"]

    return example


def update_columns_chia(example):
    global SPANS_DROPPED

    del example["text_type"]
    del example["relations"]

    pattern = r"(?<=[a-zA-Z,.])\n(?=[a-zA-Z,.])"  # the dataset is not clean, replacing \n with 2 chars to make offsets usable
    example["text"] = re.sub(pattern, "\n\n", example["text"])

    updated_entities = []
    for span_dict in example["entities"]:
        if len(span_dict["text"]) > 1:
            raise ValueError(span_dict)

        text = span_dict["text"][0]  # span_text according to dataset
        start, end = span_dict["offsets"][0][0], span_dict["offsets"][0][1]
        label = span_dict["type"]

        if (
            span_dict["text"][0] != example["text"][start:end]
        ):  # time to do a string match
            # print("mismatch in dataset")
            matches = get_all_matches(span_dict["text"][0], example["text"])
            if len(matches) != 1:
                # print(
                #     f"MULTIPLE or NO matches, will have to drop the example, {len(matches)=}"
                # )
                SPANS_DROPPED += 1
                continue
            start = matches[0][0]
            end = matches[0][1]

        updated_entities.append(
            {"text": text, "start": start, "end": end, "label": label}
        )

    example["entities"] = updated_entities

    return example


def update_columns_bc5cdr(example):
    if len(example["passages"]) > 2:
        raise ValueError(example)

    if example["passages"][0]["type"] != "title":
        raise ValueError(example, "order not fixed")

    # combine texts
    example["text"] = (
        example["passages"][0]["text"] + " " + example["passages"][1]["text"]
    )

    entities = example["passages"][0]["entities"] + example["passages"][1]["entities"]
    updated_entites = []
    for span_dict in entities:
        if (
            span_dict["text"][0]
            != example["text"][span_dict["offsets"][0][0] : span_dict["offsets"][0][1]]
        ):  # time to do a string match
            # print(f'mismatch in {example["passages"][0]["document_id"]=}')
            span_dict["offsets"][0][0] = example["text"].find(span_dict["text"][0])
            span_dict["offsets"][0][1] = span_dict["offsets"][0][0] + len(
                span_dict["text"][0]
            )
            # continue
        updated_entites.append(
            {
                "text": span_dict["text"][0],
                "start": span_dict["offsets"][0][0],
                "end": span_dict["offsets"][0][1],
                "label": span_dict["type"],
            }
        )
    example["entities"] = updated_entites
    example["document_id"] = example["passages"][0]["document_id"]
    example["id"] = example["passages"][0]["document_id"]

    del example["passages"]
    return example


def update_columns_biored(example):
    if len(example["passages"]) > 2:
        raise ValueError(example)

    if example["passages"][0]["type"] != "title":
        raise ValueError(example, "order not fixed")

    # combine texts
    example["text"] = (
        example["passages"][0]["text"][0] + " " + example["passages"][1]["text"][0]
    )

    entities = example["entities"]
    updated_entites = []
    for span_dict in entities:
        updated_entites.append(
            {
                "text": span_dict["text"][0],
                "start": span_dict["offsets"][0][0],
                "end": span_dict["offsets"][0][1],
                "label": span_dict["semantic_type_id"],
            }
        )

    example["entities"] = updated_entites
    example["document_id"] = example["pmid"]
    example["id"] = example["pmid"]

    del example["passages"]
    del example["relations"]
    del example["pmid"]
    return example


def update_columns_ncbi(example):
    global SPANS_DROPPED
    # combine texts
    example["text"] = (
        example["title"] + " " + example["abstract"]
    )  # ncbi expects a char b/w title and abstract

    entities = example["mentions"]
    updated_entites = []
    for span_dict in entities:
        if (
            span_dict["text"]
            != example["text"][span_dict["offsets"][0] : span_dict["offsets"][1]]
        ):  # time to do a string match
            # print("mismatch in dataset")
            SPANS_DROPPED += 1
            continue
            # matches = get_all_matches(span_dict["text"][0], example["text"])
            # if len(matches) != 1:
            #     print(
            #         f"MULTIPLE or NO matches, will have to drop the example, {len(matches)=}"
            #     )
            #     continue
            # start = matches[0][0]
            # end = matches[0][1]

        updated_entites.append(
            {
                "text": span_dict["text"],
                "start": span_dict["offsets"][0],
                "end": span_dict["offsets"][1],
                "label": span_dict["type"],
            }
        )

    example["entities"] = updated_entites
    example["document_id"] = example["pmid"]
    example["id"] = example["pmid"]

    del example["title"]
    del example["abstract"]
    del example["pmid"]
    del example["mentions"]

    return example


def get_int_label_map_from_dataset(dataset, dataset_name):
    if dataset_name in MANUAL_LABEL_MAP:
        return MANUAL_LABEL_MAP[dataset_name]
    labels = dataset["train"]._info.features["ner_tags"].feature.names
    cleaned_labels = [label.strip("B-").strip("I-") for label in labels]
    return {i: label_name for i, label_name in enumerate(cleaned_labels)}


def update_columns(dataset_name, **kwargs):
    global SPANS_DROPPED
    SPANS_DROPPED = 0
    match dataset_name:
        case "chia":
            return update_columns_chia
        case "biored":
            return update_columns_biored
        case "bc5cdr":
            return update_columns_bc5cdr
        case "ncbi":
            return update_columns_ncbi
        case "bc4chemd":
            return partial(
                update_example_from_tokens_and_labels,
                int_label_name_map=kwargs["int_label_name_map"],
            )
        case "jnlpba":
            return partial(
                update_example_from_tokens_and_labels,
                int_label_name_map=kwargs["int_label_name_map"],
            )
        case "bc2gm":
            return partial(
                update_example_from_tokens_and_labels,
                int_label_name_map=kwargs["int_label_name_map"],
            )

    raise ValueError(
        f"update columns function has not been implemented for the dataset, {dataset_name}"
    )


def split_dataset_from_train(dataset_to_split):
    """current implementation expects only train split
    returns 0.8, 0.1, 0.1::train, test, validation splits
    """
    dataset_to_split = dataset_to_split["train"].train_test_split(
        test_size=0.2, seed=42
    )
    dataset_to_split_test_val = dataset_to_split["test"].train_test_split(
        test_size=0.5, seed=42
    )

    del dataset_to_split["test"]

    dataset_to_split["test"] = dataset_to_split_test_val["test"]
    dataset_to_split["validation"] = dataset_to_split_test_val["train"]
    return dataset_to_split


def split_dataset_from_val(dataset_to_split):
    """current implementation expects only train & val split
    The val is split into 50 50 to get test and val
    returns train, test, validation splits
    """

    dataset_to_split_test_val = dataset_to_split["validation"].train_test_split(
        test_size=0.5, seed=42
    )

    del dataset_to_split["validation"]

    dataset_to_split["test"] = dataset_to_split_test_val["test"]
    dataset_to_split["validation"] = dataset_to_split_test_val["train"]
    return dataset_to_split


def split_dataset(dataset_to_split):
    splits_in_dataset = list(dataset_to_split.keys())
    if set(splits_in_dataset) == {"validation", "train"}:
        return split_dataset_from_val(dataset_to_split)
    elif set(splits_in_dataset) == {"train"}:
        return split_dataset_from_train(dataset_to_split)


def sanitize_dataset(
    dataset, spilts=["test", "validation", "train"], atleast_one_entity=False
):
    """returns santized dataset,
    empty text/ entites colums are dropped
    """
    dataset.set_format("pandas")

    for split in spilts:
        split_df = dataset[split][:]
        initial_rows = split_df.shape[0]
        if atleast_one_entity:
            split_df = split_df[split_df.entities.map(len).gt(0)]
        split_df = split_df[~split_df.text.map(clean_text).str.fullmatch("")]
        split_dataset = Dataset.from_pandas(split_df, preserve_index=False)
        dataset[split] = split_dataset
        # print(
        #     f"SANITIZATION: Rows dropped in {split} = {initial_rows-split_df.shape[0]}"
        # )

    return dataset


def process_dataset_splits(
    dataset_with_splits,
    processing_function: Callable,
    splits: List[str] = ["train", "test", "validation"],
):
    for split in splits:
        dataset_with_splits[split] = dataset_with_splits[split].map(processing_function)

    return dataset_with_splits


def standardize_dataset(
    hf_identifier: str,
    dataset_name: str,
    split_needed: bool,
    token_mapping_needed: bool,
):
    splits = ["train", "test", "validation"]

    dataset = load_dataset(hf_identifier)
    # print("starting standardization\n", dataset)

    if split_needed:
        dataset = split_dataset(dataset)

    if token_mapping_needed:
        int_label_name_map = get_int_label_map_from_dataset(dataset, dataset_name)
    else:
        int_label_name_map = None

    for split in splits:
        dataset[split] = dataset[split].map(
            update_columns(dataset_name, int_label_name_map=int_label_name_map)
        )
    # print(f"During standardization, {SPANS_DROPPED=}")

    return sanitize_dataset(dataset)


def update_label(row, label_map):
    spans = row["entities"]
    updated_spans = []
    dropped_labels = []
    for span in spans:
        updated_label = label_map.get(span["label"], None)
        if updated_label is None:
            dropped_labels.append(span["label"])
            continue
        span["label"] = updated_label

        updated_spans.append(span)

    row["entities"] = updated_spans
    row["dropped_labels"] = dropped_labels
    return row


def update_labels_in_splits(dataset, splits, label_map) -> Dataset:
    dataset.set_format("pandas")

    for split in splits:
        # print(f"Processing split {split}")
        split_df = dataset[split][0:]
        split_df = split_df.apply(lambda row: update_label(row, label_map), axis=1)
        # print(
        #     f"the labels {split_df.dropped_labels.explode().unique()} were dropped from {split}"
        # )

        # dropping rows/documents with no relevant/normalized spans
        split_df = split_df[~split_df.entities.map(len).eq(0)]

        split_df = split_df.drop(columns=["dropped_labels"])
        split_dataset = Dataset.from_pandas(split_df)
        dataset[split] = split_dataset
    return dataset


def load_and_prepare_dataset(dataset_identifier: str):

    dataset_identifier = dataset_identifier.lower()
    dataset_config = DATASET_INFO_MAP[dataset_identifier]

    standardized_dataset = standardize_dataset(dataset_config["hf_identifier"],
                                               dataset_identifier,
                                               dataset_config["split_needed"],
                                               dataset_config["token_mapping_needed"])
    
    # normalizing the entity type labels in the dataset and dropping un-normalized entity spans
    normalized_dataset = update_labels_in_splits(
        standardized_dataset, SPLITS, dataset_config["label_map"]
    )

    return normalized_dataset
    

# def main(datasets_to_standardize, verify=True):
#     for dataset_name, config in datasets_to_standardize.items():
#         print("* " * 8, dataset_name)
#         standardized_dataset = standardize_dataset(
#             config["hf_identifier"],
#             dataset_name,
#             config["split_needed"],
#             config["token_mapping_needed"],
#         )

#         if verify:
#             verify_dataset(standardized_dataset)

#         print("standardization complete\n", standardized_dataset)
#         standardized_dataset.save_to_disk(
#             f"../../data/standardized_datasets/{dataset_name}/"
#         )

#     pass


if __name__ == "__main__":
    pass
    # DATASET_INFO_MAP
