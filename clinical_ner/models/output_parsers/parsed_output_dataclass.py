from dataclasses import dataclass
from enum import Enum
from typing import Any

from clinical_ner.models.span_dataclasses import NERSpan


class ParsingErrorStep(Enum):
    # the generated text doesn't have the right python object
    PythonParsing = "python_parsing"

    # the parsed output doesn't have the right details in the python object
    NERParsing = "ner_parsing"


@dataclass
class ParsingReport:
    parsing_error_step: str = None
    parsing_error_message: str = None
    hallucination_counts: int = 0


@dataclass
class ParsedPythonObject:
    parsed_output: Any  # the python object after parsing the generated text
    generated_text: str
    parsing_success: bool = True
    parsing_error_message: str = None


@dataclass
class IntermediateOutputs:
    parsed_output: Any
    generated_text: str


@dataclass
class ParsedNEROutput:
    ner_spans: list[NERSpan]
    intermediate_outputs: ParsedPythonObject
    parsing_report: ParsingReport
