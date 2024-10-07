import ast
import re

from clinical_ner.models.span_dataclasses import NERSpan
from clinical_ner.models.utils import (
    get_all_matches,
)

from .parsed_output_dataclass import (
    # IntermediateOutputs,
    ParsedNEROutput,
    ParsedPythonObject,
    ParsingErrorStep,
    ParsingReport,
)


def parse_generated_list(generated_text: str) -> ParsedPythonObject:
    """
    returns a python list from a string representation of a list

    Ensure the datatype of the elements of the list?
    """
    # let's try re first
    pattern = r"'([^']*?)'(?:,|])"
    parsed_list = re.findall(pattern, generated_text)

    parsing_success = True
    original_generated_text = generated_text

    if parsed_list:
        return ParsedPythonObject(
            parsed_output=parsed_list,
            generated_text=original_generated_text,
            parsing_success=parsing_success,
        )
        # return parsed_list  # early exit, function getting ugly

    try:
        parsed_list = ast.literal_eval(generated_text)
    except Exception:  # could fail due to syntax error or name error
        print(f"Failed parsing, {generated_text=}\nRetrying with formatting")

        # Trying to fix the generated text
        if "'" not in generated_text:  # maybe the entities are not enclosed in quotes
            generated_text = generated_text.replace(", ", "', '")
            generated_text = generated_text.replace("[", "['")
            generated_text = generated_text.replace("]", "']")

        try:
            parsed_list = ast.literal_eval(generated_text)
        except Exception:  # could fail due to syntax error or name error
            print(f"Failed parsing w/ quotes fix, \n{generated_text=}")

            # maybe there are extra comments with the list
            list_string = generated_text.rsplit("\n[", 1)[-1]
            list_string = list_string.split("]\n")[0]
            list_string = "[" + list_string + "]"

            try:
                parsed_list = ast.literal_eval(generated_text)
            except Exception:
                print(f"Failed parsing w/ newline fix, \n{generated_text=}")
                parsing_success = False
                parsed_list = []

    return ParsedPythonObject(
        parsed_output=parsed_list,
        generated_text=original_generated_text,
        parsing_success=parsing_success,
    )


def parse_from_list_of_span_texts(
    generated_text, document_text, entity, normalized_label
) -> ParsedNEROutput:
    # Parsing function
    parsed_python_object = parse_generated_list(generated_text)

    if not parsed_python_object.parsing_success:
        return ParsedNEROutput(
            ner_spans=[],
            intermediate_outputs=parsed_python_object,
            parsing_report=ParsingReport(
                parsing_error_step=ParsingErrorStep.PythonParsing,
                parsing_error_message=parsed_python_object.parsing_error_message,
            ),
        )

    if normalized_label is None:
        normalized_label = entity

    parsed_python_list = parsed_python_object.parsed_output
    hallucination_counts = 0

    ner_spans = []
    for span_text in parsed_python_list:
        try:
            matches = get_all_matches(span_text, document_text)
        except Exception as e:
            print(f"{span_text=}\n{document_text=}\n{e}")
            matches = []

        if not matches:
            print(f"Entity not found, {span_text=}\n{document_text=}")
            hallucination_counts += 1

        for span_start, span_end in matches:
            ner_spans.append(
                NERSpan(
                    start=span_start,
                    end=span_end,
                    span_text=span_text,
                    label=normalized_label,
                    confidence=-1,
                )
            )
    return ParsedNEROutput(
        ner_spans=ner_spans,
        intermediate_outputs=parsed_python_object,
        parsing_report=ParsingReport(hallucination_counts=hallucination_counts),
    )
