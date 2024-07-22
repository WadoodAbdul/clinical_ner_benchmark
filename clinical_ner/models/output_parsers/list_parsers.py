import re

from langchain_core.output_parsers.list import (
    ListOutputParser,
)
from clinical_ner.models.utils import (
    get_all_matches,
)
from clinical_ner.models.span_dataclasses import NERSpan, NERSpans


def parse_generated_list(generated_text):
    """returns a python list from a string representation of a list"""
    # let's try re first
    pattern = r"'([^']*?)'(?:,|])"
    parsed_list = re.findall(pattern, generated_text)
    if parsed_list:
        return parsed_list  # early exit, function getting ugly

    try:
        parsed_list = eval(generated_text)
    except Exception:  # could fail due to syntax error or name error
        print(f"Failed parsing, {generated_text=}\nRetrying with formatting")

        # Trying to fix the generated text
        if "'" not in generated_text:  # maybe the entities are not enclosed in quotes
            generated_text = generated_text.replace(", ", "', '")
            generated_text = generated_text.replace("[", "['")
            generated_text = generated_text.replace("]", "']")

        try:
            parsed_list = eval(generated_text)
        except Exception:  # could fail due to syntax error or name error
            print(f"Failed parsing w/ quotes fix, \n{generated_text=}")

            # maybe there are extra comments with the list
            list_string = generated_text.rsplit("\n[", 1)[-1]
            list_string = list_string.split("]\n")[0]
            list_string = "[" + list_string + "]"

            try:
                parsed_list = eval(generated_text)
            except Exception:
                print(f"Failed parsing w/ newline fix, \n{generated_text=}")

                parsed_list = []

    return parsed_list

def parse_from_list_of_span_texts(generated_text,document_text, entity, normalized_label) -> list[NERSpan]:
    # Parsing function
    identified_span_texts = parse_generated_list(generated_text)

    if normalized_label is None:
        normalized_label = entity

    ner_spans = []
    for span_text in identified_span_texts:
        try:
            matches = get_all_matches(span_text, document_text)
        except Exception as e:
            print(f"{span_text=}\n{document_text=}\n{e}")
            matches = []

        if not matches:
            print(f"Entity not found, {span_text=}\n{document_text=}")
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
    return ner_spans

class PythonListOutputParser(ListOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "output_parsers", "list"]

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a list of comma separated values, sandwiched between square brackets "
            """eg: '["a", "b"]' or `['substring', 'ag']`"""
        )

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call."""
        return parse_generated_list(text)

    @property
    def _type(self) -> str:
        return "python-list-separated-by-commas"

