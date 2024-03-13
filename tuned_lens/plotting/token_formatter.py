"""Contains a class for formatting tokens for display in plots."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .utf8_utils import gpt2_token_to_bytes, safe_decode


@dataclass
class TokenFormatter:
    """Format tokens for display in a plots."""

    ellipsis: str = "…"
    newline_replacement: str = "\\n"
    newline_token: str = "Ċ"
    whitespace_token: str = "Ġ"
    whitespace_replacement: str = "_"
    max_string_len: Optional[int] = 7

    def __post_init__(self) -> None:
        """Post init hook to vectorize the format function."""
        self.vectorized_format = np.vectorize(self.format)

    def format(self, token: str) -> str:
        """Format a token for display in a plot."""
        if not isinstance(token, str):
            return "<unk>"

        if self.max_string_len is not None and len(token) > self.max_string_len:
            token = token[: self.max_string_len - len(self.ellipsis)] + self.ellipsis
        token = token.replace(self.newline_token, self.newline_replacement)
        token = token.replace(self.whitespace_token, self.whitespace_replacement)
        return token

    def pad_token_repr_to_max_len(self, token_repr: str) -> str:
        """Pad a token representation to the max string length."""
        if self.max_string_len is None:
            return token_repr
        return token_repr[: self.max_string_len] + " " * (
            self.max_string_len - len(token_repr)
        )


@dataclass
class TokenFormatterUTF8(TokenFormatter):
    """Format tokens without GPT-glyphs."""

    def format(self, token: str) -> str:
        """Format a token for display in a plot."""
        if not isinstance(token, str):
            return "<unk>"

        token_repr = safe_decode(
            gpt2_token_to_bytes(token),
            mode=None,
            replacements={
                "\n": self.newline_replacement,
                " ": self.whitespace_replacement,
            },
            forbidden_trailing=[self.ellipsis],
        )

        if self.max_string_len is not None and len(token_repr) > self.max_string_len:
            token_repr = (
                token_repr[: self.max_string_len - len(self.ellipsis)] + self.ellipsis
            )
        return token_repr
