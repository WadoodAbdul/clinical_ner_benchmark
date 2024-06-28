"""
get models in leaderboard - from pretrained list

reproduce leaderboard results(model)
    loads the config from saved config

"""
import json

from clinical_ner.models import SpanExtractor, IDENTIFIER_TO_SPAN_EXTRACTOR_MAP
from clinical_ner.evaluation import Evaluator
from clinical_ner.benchmarks import MEDICS_NER

INFERENCE_CONFIGS_PATH = '../data/inputs/inference_configs'


def get_leaderboard_models():
    return list(IDENTIFIER_TO_SPAN_EXTRACTOR_MAP.keys())


def load_leaderboard_model_and_config(model_name):
    model = SpanExtractor.from_predefined(model_name)
    with open(f'{INFERENCE_CONFIGS_PATH}/{model_name}/config.json', 'r') as f:
        config = json.load(f)
    return model, config

def reproduce_leaderboard_results(models:list| None = None, 
                                  output_dir:str|None = None):
    """
    Reproduces the results on the NCER leaderboard and stores the metrics in output_dir
    """
    if models is None:
        models = get_leaderboard_models()
    benchmark = MEDICS_NER

    for model_name in models:
        model, config = load_leaderboard_model_and_config(model_name)

        evaluator = Evaluator(model, 
                              benchmark=benchmark, 
                              dataset_wise_config=config, 
                              output_dir=output_dir)
        evaluator.run()