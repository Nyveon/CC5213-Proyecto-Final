from unidecode import unidecode


def normalize(text: str) -> str:
    """Normaliza texto

    Args:
        text (str): Texto a normalizar

    Returns:
        str: Texto normalizado
    """
    return unidecode(text.strip().lower())
