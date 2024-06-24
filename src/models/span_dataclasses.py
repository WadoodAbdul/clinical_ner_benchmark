from pydantic.dataclasses import dataclass


@dataclass
class NERSpan:
    start: int
    end: int
    span_text: str
    label: str
    confidence: float
    # parent_text: str

    def _is_superset_of(self, other_span, ensure_same_label: bool = False) -> bool:
        """returns true if the other_span's char_indice_range is a subset of or equal to the original span else false
        Arg ensure_same_label also takes label into account
        """
        if ensure_same_label:
            if self.label != other_span.label:
                return False

        if self.start <= other_span.start and self.end >= other_span.end:
            return True

        return False

    def update_offset(self, offset):
        self.start += offset
        self.end += offset


@dataclass
class NERSpans:
    parent_text: str
    spans: list[NERSpan]
