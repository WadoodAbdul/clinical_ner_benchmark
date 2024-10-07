from typing import Any, Union

from pydantic.dataclasses import dataclass

from .span_dataclasses import NERSpans


@dataclass
class IntermediateOutputs:
    model_io: list
    entities: list
    label_normalization_map: Union[dict, None]


@dataclass
class TextGenerationOutput:
    model_input: Any
    generated_text: str


@dataclass
class NERChunkOutput:
    ner_spans: NERSpans
    intermediate_outputs: IntermediateOutputs


@dataclass
class NEROutput:
    ner_spans: NERSpans
    text_chunks: list[str]
    chunkwise_intermediate_outputs: list[IntermediateOutputs]
