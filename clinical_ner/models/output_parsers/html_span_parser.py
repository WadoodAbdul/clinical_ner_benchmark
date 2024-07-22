import re

from clinical_ner.models.span_dataclasses import NERSpan, NERSpans


def parse_from_html_spans(input_text:str, document_text:str) -> list[NERSpan]:
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
        output.append(NERSpan(
                    start=start_index,
                    end=start_index+len(span_text),
                    span_text=span_text,
                    label=span_class,
                    confidence=-1,
                ))

    # Append any remaining text after the last match
    # cleaned_text += input_text[last_end:]

    return output


