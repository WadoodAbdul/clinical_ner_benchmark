from .hf_extractor import HFDecoderSpanExtractor, HFEncoderSpanExtractor

# from gliner_extractor_extractor import GLIEncoderSpanExtractor, GLITokenEncoderSpanExtractor

# current options are gliner_based, hf_encoder_based, hf_decoder_based, gliner_token_based
IDENTIFIER_TO_SPAN_EXTRACTOR_MAP = {
    "urchade/gliner_large_bio-v0.2": "gliner_based",
    "urchade/gliner_large_bio-v0.1": "gliner_based",
    "urchade/gliner_large-v2": "gliner_based",
    "alvaroalon2/biobert_diseases_ner": "hf_encoder_based",
    "bioformers/bioformer-8L-ncbi-disease": "hf_encoder_based",
    "numind/NuNER_Zero": "gliner_based",
    "/nfs/projects/healthcare/USERS/wadood/AMC/medical_coding/models/Universal-NER/UniNER-7B-type": "hf_decoder_based",
    "Meta-Llama-3-70b-Instruct": "m2_decoder_based",
    "Mixtral-8x7B-Instruct-v0.1": "m2_decoder_based",
    "numind/NuNER_Zero-span": "gliner_based",
    "urchade/gliner_large-v2.1": "gliner_based",
    "knowledgator/gliner-multitask-large-v0.5": "gliner_based",
    "gliner-community/gliner_large-v2.5": "gliner_token_based",
    "EmergentMethods/gliner_large_news-v2.1": "gliner_based",
}


def load_ner_processor(identifier, **model_kwargs):
    match IDENTIFIER_TO_SPAN_EXTRACTOR_MAP[identifier]:
        # case "gliner_based":
        #     ner_processor = GLIEncoderSpanExtractor(
        #         model_config["identifier"], **model_config["model_args"]
        #     )
        #     print(f"Loaded GLiNER model {model_config['identifier']}")
        # case "gli_ner_token_based":
        #     ner_processor = GLITokenEncoderSpanExtractor(
        #         model_config["identifier"], **model_config["model_args"]
        #     )
        #     print(f"Loaded GLiNER Token model {model_config['identifier']}")
        case "hf_encoder_based":
            ner_processor = HFEncoderSpanExtractor(identifier, **model_kwargs)
            print(f"Loaded HF NER model {identifier=} with {model_kwargs=}")
        # case "hf_decoder_based":
        #     ner_processor = HFDecoderSpanExtractor(
        #         model_config["identifier"], **model_config["model_args"]
        #     )
        #     print(f"Loaded Decoder model {model_config['identifier']}")
        # case "m2_decoder_based":
        #     ner_processor = M2SpanExtractor(
        #         model_config["identifier"], **model_config["model_args"]
        #     )
        #     print(f"Loaded M2 Decoder model {model_config['identifier']}")
        case _:
            print(f"No model {identifier=} in map")
    return ner_processor
