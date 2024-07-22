from enum import Enum

from clinical_ner.models.output_parsers import (
    parse_from_list_of_span_texts,
    parse_from_html_spans,
    PythonListOutputParser, 
    JsonOutputParser
                                                
)


def output_parser_loader(parser_identifier):
    match parser_identifier:
        case "python_list_parser":
            return parse_from_list_of_span_texts
        case "html_span_parser":
            return parse_from_html_spans
        case "json_parser":
            return JsonOutputParser
        case _:
            raise NotImplementedError(f"the parsing function {parser_identifier=} has not been implemented yet")




