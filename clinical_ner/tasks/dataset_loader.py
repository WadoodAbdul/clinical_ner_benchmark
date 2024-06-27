from .ner import CHIA, NCBI


def load_task_dataset(identifier, **kwargs):
    match identifier:
        case "NCBI":
            return NCBI("NCBI", **kwargs)
        case "CHIA":
            return CHIA("CHIA", **kwargs)
        # case "BIORED":
        #     return BIORED("BIORED", **kwargs)
        # case "BC5CDR":
        #     return BC5CDR("BC5CDR", **kwargs)
        case _:
            raise NotImplementedError(
                f"dataset with {identifier=} has not been defined yet"
            )
