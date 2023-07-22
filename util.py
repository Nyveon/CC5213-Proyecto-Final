import os

from unidecode import unidecode


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
transcripts = f"{script_dir}/videos/transcripciones/transcripcion_completa"
transcripts_json = f"{script_dir}/videos/transcripciones/transcripcion_json"


def normalize(text: str) -> str:
    """Normaliza texto

    Args:
        text (str): Texto a normalizar

    Returns:
        str: Texto normalizado
    """
    return unidecode(text.strip().lower())
