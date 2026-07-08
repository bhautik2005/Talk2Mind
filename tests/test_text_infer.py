import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "talk2mind_app", "backend"))

from inference import text_infer


def test_text_model_can_load():
    text_infer._model = None
    text_infer._vectorizer = None
    text_infer._load()
    assert text_infer._model is not None
    assert text_infer._vectorizer is not None
