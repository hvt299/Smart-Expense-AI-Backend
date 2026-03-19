import re
from underthesea import word_tokenize

def tokenize_vn(text):
    return word_tokenize(text, format="text")

def clean_text(text):
    text = re.sub(r"\d+", "", text)
    return text.strip()