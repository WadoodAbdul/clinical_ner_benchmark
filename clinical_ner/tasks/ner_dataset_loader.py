from .ner import CHIA, NCBI, BIORED, BC5CDR


def load_task_dataset(identifier, **kwargs):
    match identifier:
        case "NCBI":
            return NCBI("NCBI", **kwargs)
        case "CHIA":
            return CHIA("CHIA", **kwargs)
        case "BIORED":
            return BIORED("BIORED", **kwargs)
        case "BC5CDR":
            return BC5CDR("BC5CDR", **kwargs)
        case "NCBI_PROMPT_ENGINEERING":
            return NCBI("NCBI", **kwargs)
        case "CHIA_PROMPT_ENGINEERING":
            return CHIA("CHIA", **kwargs)
        case "BIORED_PROMPT_ENGINEERING":
            return BIORED("BIORED", **kwargs)
        case "BC5CDR_PROMPT_ENGINEERING":
            return BC5CDR("BC5CDR", **kwargs) 
        case _:
            raise NotImplementedError(
                f"dataset with {identifier=} has not been defined yet"
            )
