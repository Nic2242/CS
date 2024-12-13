# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:25:31 2024

@author: 531725ns
"""

import re

MODEL_WORD_PATTERN = r'[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*'
VALUE_MODEL_WORD_PATTERN = r'^\d+(\.\d+)?[a-zA-Z]+$|^\d+(\.\d+)?$'

# Dictionary for normalization
NORMALIZATION_MAP = {
    "inch": [ " inches", '"', "-inch", " inch", "inches"],
    "hz": ["hertz", "-hz", " hz", "-hertz", " hertz"]
}

# Create a reverse map for lookup
NORMALIZATION_REVERSE_MAP = {
    variant: key for key, variants in NORMALIZATION_MAP.items() for variant in variants
}


def normalize_text(text):
    """
    Normalize a title or value by replacing substrings matching the normalization map.
    """
    for variant, normalized in NORMALIZATION_REVERSE_MAP.items():
        if variant in text.lower():
            text = re.sub(re.escape(variant), normalized, text, flags=re.IGNORECASE)
    return text

def clean_token(token):
    """
    Remove brackets from a token and ensure that everything after 'inch' is removed if present.
    Optionally truncate tokens to their numeric part based on the provided mode.
    """
    # Remove unwanted punctuation
    token = re.sub(r'[()\[\]{};:,]', '', token)
    
    # If the token contains "inch", truncate everything after it
    if "inch" in token:
        token = token.split("inch")[0] + "inch"
    
    # # If it's a value model word and truncation is enabled, process according to the mode
    # if is_value_model_word:
    #     if truncate_all: #base case
    #         # Truncate to just the numeric part (including decimals)
    #         token = re.sub(r'[^0-9.]', '', token)
    #     else: #New case
    #         # Truncate to numeric part except for "hz", "inch", and "p"
    #         non_numeric_part = re.sub(r'[^a-zA-Z]', '', token).lower()
    #         if non_numeric_part not in {'inch','hz','p'}:
    #             token = re.sub(r'[^0-9.]', '', token)  # Keep only digits and decimal if non-numeric part isn't 'hz', 'inch', or 'p'
    
    return token

def is_model_word(text):
    """
    Check if the given text matches the model word pattern.
    """
    return re.match(MODEL_WORD_PATTERN, text) is not None


def is_model_word_value(text):
    """
    Check if the given text matches the model word value pattern.
    """
    return re.match(VALUE_MODEL_WORD_PATTERN, text) is not None


def get_model_words_title(text):
    """
    Normalize the title, then extract and return model words.
    """
    normalized_text = normalize_text(text)  # Normalize the text first
    tokens = normalized_text.lower().split()  # Tokenize the normalized text
    model_words = {clean_token(token) for token in tokens if is_model_word(token)}  # Extract model words
    return model_words


def get_model_words_value(text):
    """
    Normalize the value, then extract and return model words.
    """
    normalized_text = normalize_text(text)  # Normalize the text first
    tokens = normalized_text.lower().split()  # Tokenize the normalized text
    
    model_words = set()  # To store valid model words
    for token in tokens:
        # Match the general number pattern
        match = re.match(r'^(\d+(\.\d+)?)([a-zA-Z]*)$', clean_token(token))
        if not match:
            continue  # Skip tokens that don't match the pattern
        
        numeric_part = match.group(1)  # Extract the numeric portion
        non_numeric_part = match.group(3).lower()  # Extract the suffix
        
        if non_numeric_part:  # If there's a suffix
            if non_numeric_part in {'inch', 'hz', 'p'}:
                model_words.add(clean_token(token))  # Add the full token if the suffix is valid
            elif '.' in numeric_part:  # For invalid suffixes, keep the numeric part if it's a decimal
                model_words.add(numeric_part)
        else:  # No suffix: include only if the numeric part is a decimal
            if '.' in numeric_part:
                model_words.add(numeric_part)
    
    return model_words
    

