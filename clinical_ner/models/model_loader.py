from .hf_extractor import HFDecoderSpanExtractor, HFEncoderSpanExtractor
from .llm_api_extractor import LLMSpanExtractor
from .gliner_extractor import GLiNEREncoderSpanExtractor

# current options are gliner_based, hf_encoder_based, hf_decoder_based, gliner_token_based
IDENTIFIER_TO_SPAN_EXTRACTOR_MAP = {
    "urchade/gliner_large_bio-v0.2": "gliner_based",
    "urchade/gliner_large_bio-v0.1": "gliner_based",
    "urchade/gliner_large-v2": "gliner_based",
    "alvaroalon2/biobert_diseases_ner": "hf_encoder_based",
    "bioformers/bioformer-8L-ncbi-disease": "hf_encoder_based",
    "numind/NuNER_Zero": "gliner_based",
    "Universal-NER/UniNER-7B-type": "hf_decoder_based",
    "Meta-Llama-3-70b-Instruct": "llm_api_decoder_based",
    "Mixtral-8x7B-Instruct-v0.1": "llm_api_decoder_based",
    "numind/NuNER_Zero-span": "gliner_based",
    "urchade/gliner_large-v2.1": "gliner_based",
    "knowledgator/gliner-multitask-large-v0.5": "gliner_based",
    "gliner-community/gliner_large-v2.5": "gliner_token_based",
    "EmergentMethods/gliner_large_news-v2.1": "gliner_based",
}


def load_ner_processor(identifier, **model_kwargs):
    match IDENTIFIER_TO_SPAN_EXTRACTOR_MAP[identifier]:
        case "gliner_based":
            ner_processor = GLiNEREncoderSpanExtractor(identifier, **model_kwargs)
            print(f"Loaded GLiNER model {identifier}")
        case "gliner_token_based":
            ner_processor = GLiNEREncoderSpanExtractor(identifier, is_token_based=True, **model_kwargs)
            print(f"Loaded GLiNER model {identifier}")
        case "hf_encoder_based":
            ner_processor = HFEncoderSpanExtractor(identifier, **model_kwargs)
            print(f"Loaded HF Encoder NER model {identifier=} with {model_kwargs=}")
        case "hf_decoder_based":
            ner_processor = HFDecoderSpanExtractor(identifier, **model_kwargs)
            print(f"Loaded HF Decoder NER model {identifier=} with {model_kwargs=}")
        case "llm_api_decoder_based":
            ner_processor = LLMSpanExtractor(identifier, **model_kwargs)
            print(f"Loaded API LLM Decoder model {identifier}")
        case _:
            print(f"No model {identifier=} in map")
    return ner_processor
