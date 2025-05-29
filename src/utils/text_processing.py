import re
import numpy as np

def normalize(v):
    """Normalize vector"""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def preprocess_text(text):
    """Tiền xử lý text để tách các từ dính liền nhau"""
    text = re.sub(r'([a-zàáảãạâầấẩẫậêềếểễệôồốổỗộưừứửữựđ])([A-ZÀÁẢÃẠÂẦẤẨẪẬÊỀẾỂỄỆÔỒỐỔỖỘƯỪỨỬỮỰĐ])', r'\1 \2', text)
    text = re.sub(r'([a-zàáảãạâầấẩẫậêềếểễệôồốổỗộưừứửữựđ])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zàáảãạâầấẩẫậêềếểễệôồốổỗộưừứửữựđA-ZÀÁẢÃẠÂẦẤẨẪẬÊỀẾỂỄỆÔỒỐỔỖỘƯỪỨỬỮỰĐ])', r'\1 \2', text)
    return text 