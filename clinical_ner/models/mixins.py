import re
from typing import Generator, Optional, Union

from .span_dataclasses import NERSpans
from .utils import (
    get_char_label_map,
    get_list_of_token_label_tuples,
)


class WhitespaceTokenSplitter:
    def __init__(self):
        self.whitespace_pattern = re.compile(r"\w+(?:[-_]\w+)*|\S")

    def __call__(self, text) -> Generator[str, int, int]:
        for match in self.whitespace_pattern.finditer(text):
            yield match.group(), match.start(), match.end()


class WhiteSpaceTokenizerMixin:
    def __init__(self) -> None:
        self.tokenizer = WhitespaceTokenSplitter()

    def get_tokens(self, text: str) -> list[str]:
        tokens_with_offsets = list(self.tokenizer(text))
        return [token for token, start, end in tokens_with_offsets]

    def get_token_offsets(self, text: str) -> list[tuple[int, int]]:
        tokens_with_offsets = list(self.tokenizer(text))
        return [(start, end) for token, start, end in tokens_with_offsets]

    def get_token_level_ner_output_from_spans(
        self, ner_spans: Union[list[dict], NERSpans], parent_text: Optional[str] = None
    ) -> list[str]:
        """returns a list of tuples of token and label
        example - [('a', 'O'), ('cat', 'ANIMAL'), ('sits', 'O')]
        the tokens with no labels are assigned to 'O'

        Useful for token level evaluation!
        """
        if not parent_text:  # in the case where model prediction/NERSpans is passed - can be solved if dataclass is subscriptable
            parent_text = ner_spans.parent_text
            ner_spans = [span.__dict__ for span in ner_spans.spans]

        char_label_map = get_char_label_map(ner_spans)

        token_offsets = self.get_token_offsets(parent_text)
        tokens = self.get_tokens(parent_text)

        return get_list_of_token_label_tuples(tokens, token_offsets, char_label_map)
