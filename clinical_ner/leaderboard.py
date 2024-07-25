"""
get models in leaderboard - from pretrained list

reproduce leaderboard results(model)
    loads the config from saved config

"""
import json

from clinical_ner.models import IDENTIFIER_TO_SPAN_EXTRACTOR_MAP
from clinical_ner.evaluation import Evaluator
from clinical_ner.benchmarks import MEDICS_NER
from clinical_ner.models.gliner_extractor import GLiNEREncoderSpanExtractor
from clinical_ner.models.hf_extractor import HFEncoderSpanExtractor, HFDecoderSpanExtractor

INFERENCE_CONFIGS_PATH = '../data/inputs/inference_configs'


def get_leaderboard_models():
    return list(IDENTIFIER_TO_SPAN_EXTRACTOR_MAP.keys())

def make_dataset_wise_config(config):
    leaderboard_datasets = ['NCBI', 'CHIA', 'BIORED', 'BC5CDR']


    inference_config = config['inference_config']

    # if request_row.status == "RERUN":
    #     return inference_config

    dataset_wise_config = {}
    for dataset in leaderboard_datasets:
        dataset_config = {
            "label_normalization_map": inference_config['label_normalization_map'][dataset],
            # "prompt_template_identifier": inference_config['prompt_template_identifier'],
        }

        if config['model_architecture'] == "Decoder":
            dataset_config = {
                **dataset_config,
                "prompt_template_identifier": inference_config['prompt_template_identifier'],
            }
        dataset_wise_config[dataset] = dataset_config

    return dataset_wise_config

def load_leaderboard_model_and_config(model_name):
    with open(f'{INFERENCE_CONFIGS_PATH}/{model_name}/config.json', 'r') as f:
        config = json.load(f)

    dataset_wise_config = make_dataset_wise_config(config)

    if config['model_architecture'] == "Encoder":
        model = HFEncoderSpanExtractor(config['model_name'])
    elif config['model_architecture'] == "Decoder":
        model = HFDecoderSpanExtractor(config['model_name'])
    elif config['model_architecture'] == "GLiNER Encoder":
        model = GLiNEREncoderSpanExtractor(
            config['model_name'], 
            threshold=config['inference_config']['gliner_threshold'],
            is_token_based=config['inference_config']['gliner_tokenizer_bool']
            )

    return model, dataset_wise_config

def reproduce_leaderboard_results(models:list| None = None, 
                                  output_dir:str|None = None):
    """
    Reproduces the results on the NCER leaderboard and stores the metrics in output_dir
    """
    if models is None:
        models = get_leaderboard_models()
    benchmark = MEDICS_NER

    for model_name in models:
        model, dataset_wise_config = load_leaderboard_model_and_config(model_name)

        evaluator = Evaluator(model, 
                              benchmark=benchmark, 
                              dataset_wise_config=dataset_wise_config, 
                              output_dir=output_dir)
        evaluator.run()