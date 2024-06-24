
from hf_extractor import HFEncoderSpanExtractor, HFDecoderSpanExtractor
# from gliner_extractor_extractor import GLIEncoderSpanExtractor, GLITokenEncoderSpanExtractor

IDENTIFIER_TO_SPAN_EXTRACTOR_MAP = {
    "urchade/gliner_large_bio-v0.2": "gli_ner_based",
    "urchade/gliner_large_bio-v0.1": "gli_ner_based",
    "urchade/gliner_large-v2": "gli_ner_based",
    "alvaroalon2/biobert_diseases_ner": "hf_encoder_based",
    "bioformers/bioformer-8L-ncbi-disease": "hf_encoder_based",
    "numind/NuNER_Zero": "gli_ner_based",
    "/nfs/projects/healthcare/USERS/wadood/AMC/medical_coding/models/Universal-NER/UniNER-7B-type": "hf_decoder_based",
    "Meta-Llama-3-70b-Instruct": "m2_decoder_based",
    "Mixtral-8x7B-Instruct-v0.1": "m2_decoder_based",
    "numind/NuNER_Zero-span": "gli_ner_based",
    "urchade/gliner_large-v2.1": "gli_ner_based",
    "knowledgator/gliner-multitask-large-v0.5": "gli_ner_based",
    "gliner-community/gliner_large-v2.5": "gli_ner_token_based",
    "EmergentMethods/gliner_large_news-v2.1": "gli_ner_based",
}

def load_ner_processor(model_config):
    match IDENTIFIER_TO_SPAN_EXTRACTOR_MAP[model_config["identifier"]]:
        case "gli_ner_based":
            ner_processor = GLIEncoderSpanExtractor(
                model_config["identifier"], **model_config["model_args"]
            )
            print(f"Loaded GLiNER model {model_config['identifier']}")
        case "gli_ner_token_based":
            ner_processor = GLITokenEncoderSpanExtractor(
                model_config["identifier"], **model_config["model_args"]
            )
            print(f"Loaded GLiNER Token model {model_config['identifier']}")
        case "hf_encoder_based":
            ner_processor = HFEncoderSpanExtractor(
                model_config["identifier"], **model_config["model_args"]
            )
            print(f"Loaded HF NER model {model_config['identifier']}")
        case "hf_decoder_based":
            ner_processor = HFDecoderSpanExtractor(
                model_config["identifier"], **model_config["model_args"]
            )
            print(f"Loaded Decoder model {model_config['identifier']}")
        case "m2_decoder_based":
            ner_processor = M2SpanExtractor(
                model_config["identifier"], **model_config["model_args"]
            )
            print(f"Loaded M2 Decoder model {model_config['identifier']}")
        case _:
            print(f"No model {model_config['identifier']} in map")
    return ner_processor