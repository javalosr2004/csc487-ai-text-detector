import re
import unicodedata


def normalize_whitespace(text):
    """Normalize whitespace by replacing multiple spaces with single space."""
    # Replace non-breaking spaces with regular spaces
    text = text.replace('\xa0', ' ')
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text


def normalize_unicode(text):
    """Normalize unicode characters to standard forms."""
    # Normalize unicode to NFKD form (compatibility decomposition)
    text = unicodedata.normalize('NFKD', text)
    return text


def remove_control_characters(text):
    """Remove or replace control characters."""
    # Define control characters to remove (excluding newline, tab, carriage return)
    control_chars = {
        '\x82': ',',   # Single Low-9 Quotation Mark
        '\x92': "'",   # Left Single Quotation Mark
        '\x93': '"',   # Left Double Quotation Mark
        '\x94': '"',   # Right Double Quotation Mark
        '\x97': '-',   # Em Dash
        '\x91': "'",   # Left Single Quotation Mark
        '\x96': '-',   # En Dash
        '\x85': '...',  # Horizontal Ellipsis
    }

    for char, replacement in control_chars.items():
        text = text.replace(char, replacement)

    # Remove any remaining control characters except whitespace
    text = ''.join(char for char in text if ord(char) >= 32 or char in ['\n', '\t', '\r'])

    return text


def normalize_quotes(text):
    """Normalize various quote styles to standard quotes."""
    # Smart/curly quotes to straight quotes
    quote_map = {
        '"': '"',  # Left double quotation mark
        '"': '"',  # Right double quotation mark
        ''': "'",  # Left single quotation mark
        ''': "'",  # Right single quotation mark
        '«': '"',  # Left-pointing double angle quotation mark
        '»': '"',  # Right-pointing double angle quotation mark
    }

    for fancy_quote, standard_quote in quote_map.items():
        text = text.replace(fancy_quote, standard_quote)

    return text


def remove_structural_markers(text):
    """Remove common structural markers from essays."""
    # Remove patterns like "Title:", "Introduction:", etc. at the beginning
    patterns = [
        r'^Title:\s*',
        r'^Introduction:\s*',
        r'^Conclusion:\s*',
        r'^Body Paragraph \d+:\s*',
        r'^Paragraph \d+:\s*',
    ]

    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text


def preprocess_text(text, remove_markers=False):
    """
    Apply all preprocessing steps to text.

    Args:
        text: Input text string
        remove_markers: Whether to remove structural markers (default: False)

    Returns:
        Preprocessed text string
    """
    if not isinstance(text, str):
        return ""

    # Apply preprocessing steps in order
    text = remove_control_characters(text)
    text = normalize_unicode(text)
    text = normalize_quotes(text)
    text = normalize_whitespace(text)

    if remove_markers:
        text = remove_structural_markers(text)
        text = normalize_whitespace(text)  # Re-normalize after marker removal

    return text


def filter_short_texts(texts, labels, min_length=100):
    """
    Filter out texts that are too short.

    Args:
        texts: List of text strings
        labels: List of labels
        min_length: Minimum character length (default: 100)

    Returns:
        Tuple of (filtered_texts, filtered_labels)
    """
    filtered_texts = []
    filtered_labels = []

    for text, label in zip(texts, labels):
        if len(text) >= min_length:
            filtered_texts.append(text)
            filtered_labels.append(label)

    return filtered_texts, filtered_labels
