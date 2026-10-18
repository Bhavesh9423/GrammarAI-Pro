"""Lazy, optional Hugging Face grammar-correction inference."""

import logging
import os
import threading

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "prithivida/grammar_error_correcter_v1"
_load_lock = threading.Lock()
_model = None
_tokenizer = None
_load_error = None


def model_name():
    return os.getenv("GRAMMAR_AI_MODEL", DEFAULT_MODEL)


def _load_model():
    """Load one local model instance; importing transformers stays optional."""
    global _model, _tokenizer, _load_error
    if _model is not None:
        return _model, _tokenizer
    if _load_error is not None:
        raise RuntimeError(_load_error)
    with _load_lock:
        if _model is not None:
            return _model, _tokenizer
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            name = model_name()
            _tokenizer = AutoTokenizer.from_pretrained(name)
            _model = AutoModelForSeq2SeqLM.from_pretrained(name)
            _model.eval()
            logger.info("Loaded optional grammar AI model: %s", name)
            return _model, _tokenizer
        except Exception as exc:
            _load_error = "The optional AI model could not be loaded. Check the model name, network connection, disk space, and transformers installation."
            logger.warning("Grammar AI model unavailable: %s", exc)
            raise RuntimeError(_load_error) from exc


def correct_with_ai(text: str) -> dict:
    """Generate a correction using a real seq2seq model, never a rule fallback."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text is required for AI correction.")
    if len(text) > 5000:
        raise ValueError("AI correction supports up to 5,000 characters per request.")

    model, tokenizer = _load_model()
    prompt = "gec: " + text.strip()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    generated = model.generate(**inputs, max_new_tokens=256, num_beams=4, early_stopping=True)
    corrected = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    if not corrected:
        raise RuntimeError("The AI model returned an empty correction.")
    return {"original_text": text.strip(), "corrected_text": corrected, "model": model_name()}
