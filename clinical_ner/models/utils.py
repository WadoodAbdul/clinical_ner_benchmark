import re
from pathlib import Path
from typing import Dict, Tuple


def ceiling_division(numerator, denominator):
    """
    returns the ceiling of the quotient of the division
    source: https://stackoverflow.com/a/54585138
    """
    return -(numerator // -denominator)


def read_template(file_path: str) -> str:
    """Reads and returns a template"""

    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"{file_path} is not a valid template.")

    return path.read_text()


def get_char_label_map(ner_spans: list):
    """return a dict with char indices(int) as keys and the label they belong to as values
    example -- {1:'label1', 2: 'label1', 5:'label2', 5:'label2'}
    note: the char indices that do not belong to a span do not exist in the map
    """
    char_label_map = {}
    for span in ner_spans:
        char_label_map = {
            **char_label_map,
            **{
                char_index: span["label"]
                for char_index in range(span["start"], span["end"])
            },
        }
    return char_label_map


def merge_spans_with_same_labels(span_list: list):
    """returns the list of spans by dropping subset spans
    subset spans have same label and are subset of another span in the list

    Expects a sorted list of spans, 1st key -> start index, 2nd key -> -len(span_text)

    Future implementation: add option to fuse overlapping spans
    """
    final_spans = [span_list[0]]
    for index, span in enumerate(span_list[1:]):
        if final_spans[-1]._is_superset_of(span, ensure_same_label=True):
            continue

        final_spans.append(span)

    return final_spans


def get_list_of_token_label_tuples(
    tokens: list[str],
    token_spans: list[Tuple[int, int]],
    char_label_map: Dict[int, str],
) -> list[Tuple[str, str]]:
    """
    returns a list of tuples with first element as token and second element as the label
    example - [('a', 'O'), ('cat', 'ANIMAL'), ('sits', 'O')]
    note: the label of a token is decided based on the max chars in the token belonging to a span
    """
    token_labels = []
    for token, offsets in zip(tokens, token_spans):
        if offsets[0] == offsets[1]:
            token_labels.append((token, "O"))
            continue
        char_labels = [
            char_label_map.get(char_index, "O") for char_index in range(*offsets)
        ]
        token_label = max(set(char_labels), key=char_labels.count)
        token_labels.append((token, token_label))
    return token_labels


def get_iob_stripped_entity_set(entities):
    """
    Note: this also excludes the o/0/O tag"""
    return list(
        (
            map(
                lambda label: label.strip("B-"),
                filter(lambda label: "B-" in label, entities),
            )
        )
    )


def split_text_into_parts(text, number_of_chunks) -> Tuple[list[str], list[str]]:
    """returns a tuple of lists, one with the chunks and the other with separators
    note: currently, the chunks are inclusive of the separators.
    """
    chunk_length = ceiling_division(len(text), number_of_chunks)
    chunks = []
    chars_betwen_chunks = []
    remaining_text = text

    while len(remaining_text) > chunk_length:
        text_before_threshold = remaining_text[:chunk_length]
        last_delimiter_index_threshold = text_before_threshold.rfind("\n")
        split_char = "\n"

        if last_delimiter_index_threshold == -1:
            last_delimiter_index_threshold = text_before_threshold.rfind(".")
            split_char = "."
            if last_delimiter_index_threshold == -1:
                last_delimiter_index_threshold = text_before_threshold.rfind(" ")
                split_char = " "
                if last_delimiter_index_threshold == -1:
                    last_delimiter_index_threshold = chunk_length - 1
                    split_char = ""

        first_segment = remaining_text[: last_delimiter_index_threshold + 1]
        remaining_text = remaining_text[last_delimiter_index_threshold + 1 :]
        chunks.append(first_segment)
        chars_betwen_chunks.append(split_char)

    if len(remaining_text) <= chunk_length:
        chunks.append(remaining_text)

    return chunks, chars_betwen_chunks


def get_all_matches(span_text, document_text, ignore_case=True):
    """returns a list of span matches in the form of [(start, end)]"""
    if ignore_case:
        flag = re.IGNORECASE.value
    else:
        flag = 0

    re_matches = list(re.finditer(re.escape(span_text), document_text, flag))
    return [match.span() for match in re_matches]


def get_first_match(span_text, document_text, ignore_case=True):
    """returns list of single element corresponding to the first occuring span_text in document_text.
    returns empty list if no match found
    """
    if ignore_case:
        flag = re.IGNORECASE.value
    else:
        flag = 0

    first_match = re.search(re.escape(span_text), document_text, flag)
    if first_match is None:
        return []
    return [first_match.span()]


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


def extract_html_spans(input_text:str) -> list[dict]:
    # Regular expression to find all span tags with their classes and contents
    # span_regex = re.compile(r'<span class="([^"]+)" >(.*?)</span >')
    input_text = input_text.replace("'", '"')
    span_regex = re.compile(r'<span class="([^"]+)" ?>(.*?)</span ?>')
    matches = span_regex.finditer(input_text)
    # List to store the extracted information
    output = []
    # Cleaned text to track the actual position
    cleaned_text = ""
    last_end = 0
    offset = 0
    for match in matches:
        span_class = match.group(1)
        span_text = match.group(2)
        # Append text before the span to the cleaned text
        cleaned_text += input_text[last_end:match.start()]
        # Current start index in the cleaned text
        start_index = len(cleaned_text)
        # Append span text to the cleaned text
        cleaned_text += span_text
        # Update last_end to the end of the current match
        last_end = match.end()
        output.append({
            'text': span_text,
            'start': start_index,
            'stop': start_index+len(span_text),
            'label': span_class
        })

    # Append any remaining text after the last match
    cleaned_text += input_text[last_end:]
    return output

