"""Utilities for working with UTF-8 data in token strings."""
import unicodedata2
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

gpt2_byte_encoder = bytes_to_unicode()
gpt2_byte_decoder = {v: k for k, v in gpt2_byte_encoder.items()}


def gpt2_token_to_bytes(token: str) -> bytes:
    """Convert a GPT-2 token string to bytes."""
    return bytes([gpt2_byte_decoder[char] for char in token])


def safe_decode(
    data: bytes, mode=None, replacements=None, forbidden_trailing=None
) -> str:
    """Decode a byte string as UTF-8, escaping unprintable bytes."""
    if replacements is None:
        replacements = {" ": "␣"}
    if forbidden_trailing is None:
        forbidden_trailing = []

    chunks = list(utf8_chunk_bytes(data))
    out = "".join(
        safe_decode_chunk(chunk, mode, replacements, []) for chunk in chunks[:-1]
    )
    if chunks:
        out += safe_decode_chunk(chunks[-1], mode, replacements, forbidden_trailing)

    return out


def safe_decode_chunk(chunk: bytes, mode, replacements: dict, forbidden: list) -> str:
    r"""Decode a chunk of bytes as UTF-8, escaping unprintable bytes.

    Keyword Arguments:
    mode: 'terse' or 'verbose' or None
    replacements: dict of Unicode characters to replace with alternatives
        (e.g. {' ': '␣'})
    forbidden: list of Unicode characters that must be escaped

    returns: string of Unicode characters and custom escape sequences

    This function tries to balance readability, transparency, reversibility,
     and terseness.
    mode 'verbose' is the most transparent and reversible, but also the
     longest.
    mode 'terse' is the shortest, but also the least transparent and
     reversible.
    mode None is in between, and is the default.
    mode 'braille' is like 'terse' but uses braille patterns to show bytes
    compactly.

    Note that Python's strings can't represent non-Unicode data, so I invented
     my own escape sequences.
    Differences from Python string literals:
    - arbitrary replacements are allowed (e.g. '␣' for SPACE); replacement
     characters themselves are escaped
    - \x always represents bytes, not Unicode characters U+0000..U+00FF
    - \U has 6 hex digits, not 8 (why not improve things while we're breaking
     compatibility?)
    - \u and \U are also used for invalid codepoints (e.g. surrogates)
    - \u[...] and \U[...] are used to represent partial Unicode characters

    Possible wrinkles:
    - Lack of font support may cause tofu in terse/default
    - isolated combining characters may look weird in terse/default
    - Non-UTF-8 data is treated as (attempted) UTF-8, which may cause confusion

    Escape sequences:
    U+2423 ␣ OPEN BOX (for space)
    \t, \n, \r, \f, \v (for ASCII control characters)
    U+FFFD � REPLACEMENT CHARACTER (for unprintable chars, partial chars,
     and invalid bytes)
    \xXX (for unprintable ASCII characters or invalid bytes)
    \N{...} (for named Unicode characters) (compatible with Python strings)
    \uXXXX or \UXXXXXX (for partial Unicode characters) (note that Python's \U
     has 8 hex digits and mine has 6)
    \u[XXXX-XXXX] or \U[XXXXXX-XXXXXX] (for verbose partial Unicode characters)

    Defaults:
    printable Unicode characters are represented by themselves (regardless of
     font support, etc)
    space, \t, \n, \r, \f, \v are represented by ␣, \t, \n, \r, \f, \v
    other unprintable Unicode characters are represented by their code points
     with \x, \u, or \U
    invalid bytes are represented by \xXX
    partial Unicode characters are represented by \uXXXX or \UXXXXXX with
     question marks for unknown digits

    In verbose mode:
    named (printable or unprintable) Unicode characters are represented by
     their names with \N{...}
    partial Unicode characters are represented by ranges with \u[XXXX-XXXX]
     or \U[XXXXXX-XXXXXX]
    (this is lossless, hence reversible to bytes, but I haven't written the
     reverse function yet.)

    In terse mode:
    \xXX for unprintable ASCII characters
    � for all other unprintable characters, partial characters, and
     invalid bytes

    In braille mode:
    \xXX for unprintable ASCII characters
    braille patterns for all other unprintable characters, partial characters,
     and invalid bytes
    (dots correspond to bits but in a weird order, see Wikipedia article
     "Braille Patterns")
    """
    try:
        # case: Unicode character
        # will raise UnicodeDecodeError if chunk is not a valid UTF-8 sequence
        char = chunk.decode("utf-8")

        if char in replacements:
            return replacements[char]
        if char in replacements.values():
            raise ValueError
        if char in forbidden:
            raise ValueError
        if mode == "braille" and 0x2800 <= ord(char) < 0x2900:
            raise ValueError

        if ord(char) < 0x80:
            # case: ASCII, including \n \t and other controls.
            return repr(char)[1:-1]

        # case: named Unicode character
        # will raise ValueError if char has no name (e.g. control characters)
        name = unicodedata2.name(char)

        if mode == "verbose":
            return f"\\N{{{name}}}"

        # case: non-printable Unicode character
        if unicodedata2.category(char)[0] not in "LNSM":
            raise ValueError
        # case: printable Unicode character
        return repr(char)[1:-1]

    except (UnicodeDecodeError, ValueError):
        if mode == "terse":
            # replacement character
            return "�"

        if mode == "braille":
            return "".join(chr(0x2800 + byte) for byte in chunk)

        # case: prefix of a multi-byte character (could be surrogate etc.)
        btypes = [utf8_byte_type(byte) for byte in chunk]
        if btypes[0] > 1 and all(btype == 0 for btype in btypes[1:]):
            return unicode_partial(*utf8_value_range(chunk), mode == "verbose")

        # case: invalid byte sequence
        return "".join(f"\\x{byte:02x}" for byte in chunk)


def utf8_byte_type(byte) -> int:
    """Return the number of bytes in the UTF-8 sequence starting with byte."""
    if byte & 0b10000000 == 0:
        # ASCII
        return 1
    elif byte & 0b11000000 == 0b10000000:
        # Continuation
        return 0
    elif byte & 0b11100000 == 0b11000000:
        # Start of 2-byte sequence
        return 2
    elif byte & 0b11110000 == 0b11100000:
        # Start of 3-byte sequence
        return 3
    elif byte & 0b11111000 == 0b11110000:
        # Start of 4-byte sequence
        return 4
    else:
        # Invalid
        return 1


def utf8_value_range(chunk: bytes):
    """Return the range of Unicode values represented by a partial UTF-8 sequence."""
    assert len(chunk) > 0
    btypes = [utf8_byte_type(byte) for byte in chunk]
    intended_len = btypes[0]
    assert (
        intended_len > 0
        and all(btype == 0 for btype in btypes[1:])
        and len(chunk) <= intended_len
    )

    min_val = (chunk[0] & (0b11111111 >> intended_len)) << (6 * (intended_len - 1))
    for i, byte in enumerate(chunk[1:], 1):
        min_val |= (byte & 0b00111111) << (6 * (intended_len - i - 1))

    max_val = min_val
    for i in range(intended_len - len(chunk)):
        max_val |= 0b00111111 << (6 * i)

    return min_val, max_val


def unicode_partial(min_val, max_val, verbose=False):
    """Return a string representing a partial Unicode character."""
    # unicode_partial(0x2000, 0x203f) -> '\u20??'
    # unicode_partial(0x2000, 0x203f, verbose=True) -> '\u[2000-203f]'
    if max_val > 0xFFFF:
        L, p = 6, "U"
    else:
        L, p = 4, "u"

    if min_val == max_val:
        return f"\\{p}{min_val:0{L}x}"

    if verbose:
        return f"\\{p}[{min_val:0{L}x}-{max_val:0{L}x}]"

    min_str = f"\\{p}{min_val:0{L}x}"
    max_str = f"\\{p}{max_val:0{L}x}"

    match_len = 0
    for i in range(len(min_str)):
        if min_str[i] == max_str[i]:
            match_len += 1
        else:
            break

    return min_str[:match_len] + "?" * (len(min_str) - match_len)


def utf8_chunk_bytes(data: bytes):
    """Yield chunks of bytes that form (possibly invalid) UTF-8 sequences."""
    chunk = []
    countdown = 0
    for byte in data:
        btype = utf8_byte_type(byte)
        if btype or countdown == 0:
            if chunk:
                yield bytes(chunk)
            chunk = [byte]
            countdown = max(btype - 1, 0)
        else:
            chunk.append(byte)
            countdown -= 1
    if chunk:
        yield bytes(chunk)
