"""
Buscador de videos por similitud de texto
Usando Fasttext y similitud coseno.
"""

import scipy.spatial
import fasttext
import fasttext.util
import os
import pickle  # nosec
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

from sklearn.manifold import TSNE
from typing import Callable, Dict, List

sys.path.append("..")
from util import normalize, transcripts, transcripts_json  # noqa: E402


buscador = "fasttext"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

model = None


def load_model() -> None:
    """Carga el modelo de Fasttext si no está cargado
    """
    global model

    if model is None:
        current_wd = os.getcwd()
        os.chdir(script_dir)
        fasttext.util.download_model("es", if_exists="ignore")
        fasttext_model_path = "cc.es.300.bin"
        model = fasttext.load_model(
            os.path.join(script_dir, fasttext_model_path))
        os.chdir(current_wd)


def sentence_descriptor(filename: str) -> list:
    """Función Descriptor. Extrae descriptores de fragmentos de video.

    Args:
        filename (str): Archivo de transcripción

    Returns:
        list: Vector descriptor
    """
    descriptors = []
    filename = filename.split(".txt")[0] + ".json"
    with open(os.path.join(transcripts_json, filename),
              "r", encoding="utf-8") as f:
        f.readline()
        f.readline()
        json_text = f.readline()
        json_transcript = json.loads(json_text)
        for fragment in json_transcript:
            descriptor = model.get_sentence_vector(normalize(fragment["text"]))
            descriptors.append(descriptor)

    return descriptors


def text_descriptor(filename: str) -> list:
    """Función Descriptor. Extrae descriptores de la transcripción completa.

    Args:
        filename (str): Archivo de transcripción

    Returns:
        list: Vector descriptor
    """
    descriptor = None
    with open(os.path.join(transcripts, filename),
              "r", encoding="utf-8") as f:
        f.readline()
        f.readline()
        text = f.readline()
        descriptor = model.get_sentence_vector(normalize(text))
    return descriptor


def title_descriptor(filename: str) -> list:
    """Función Descriptor. Extrae descriptores del título.

    Args:
        filename (str): Archivo de transcripción

    Returns:
        list: Vector descriptor
    """
    descriptor = None
    with open(os.path.join(transcripts, filename),
              "r", encoding="utf-8") as f:
        title = f.readline()
        descriptor = model.get_sentence_vector(normalize(title))
    return descriptor


def load_descriptors(recalc: bool,
                     f_descriptor: Callable[[str], list]) -> object:
    """Carga los descriptores pre-calculados o los calcula si no existen

    Args:
        recalc (bool): Obligar recalculo de descriptores.
        f_descriptor (f(str) -> list): Funcion de calculo de descriptores

    Returns:
        object: Descriptores
    """
    file = f"{script_dir}/{buscador}_{f_descriptor.__name__}.pkl"

    if not recalc and os.path.exists(file):
        with open(file, "rb") as f:
            return pickle.load(f)  # nosec

    print("Calculando descriptores por primera vez.")
    vectors = {}

    for filename in os.listdir(transcripts):
        vid = filename.split(".txt")[0]
        vectors[vid] = f_descriptor(filename)

    with open(file, "wb") as f:
        pickle.dump(vectors, f)

    return vectors


def buscar(texto_consulta: list, n: int, f_descriptor: Callable[[str], list],
           recalc=False) -> Dict[str, List[str]]:
    """Busca los n videos más similares a cada query usando fasttext
    Espacio de busqueda: Transcripciones completas

    Args:
        texto_consulta (list): lista de queries
        n (int): numero de resultados por query
        recalc (bool, optional): Obligar recalculo de descriptores
        f_descriptor (f(str) -> list): Funcion de calculo de descriptores

    Returns:
        Dict[str, List[str]]: {query: [video_id1, video_id2, ...]}
    """
    load_model()
    vectors = load_descriptors(recalc, f_descriptor)

    results = {}

    for q in texto_consulta:
        query = normalize(q)
        query_vector = model.get_sentence_vector(query)
        closest = None
        distances = {}
        if f_descriptor == sentence_descriptor:
            for video_id, vector_list in vectors.items():
                for vector in vector_list:
                    distance = scipy.spatial.distance.cosine(
                        query_vector, vector)
                    if video_id not in distances:
                        distances[video_id] = distance
                    else:
                        distances[video_id] = min(
                            distances[video_id], distance)
        else:
            for video_id, vector in vectors.items():
                distance = scipy.spatial.distance.cosine(query_vector, vector)
                distances[video_id] = distance
        closest = sorted(distances.items(), key=lambda x: x[1])[:n]
        results[q] = [str(x[0]) for x in closest]

    return results


def visualize(vectors: dict) -> None:
    """Visualiza los vectores en espacio 2D

    Args:
        vectors (dict): {video_id: vector}
    """

    vectors_list = np.array(list(vectors.values()))
    video_ids = list(vectors.keys())

    labels = [vid[:2] for vid in video_ids]
    unique_labels = list(set(labels))

    colormap = plt.colormaps.get_cmap('tab20')
    color_dict = {label: colormap(i) for i, label in enumerate(unique_labels)}

    tsne = TSNE(n_components=2, random_state=0)
    vectors_2d = tsne.fit_transform(vectors_list)
    plt.figure(figsize=(10, 10))

    for i, video_id in enumerate(video_ids):
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1],
                    color=color_dict[labels[i]])
        plt.annotate(video_id, (vectors_2d[i, 0], vectors_2d[i, 1]))

    patches = [plt.Line2D(
        [0], [0], marker='o', color='w', label=label,
        markerfacecolor=color,
        markersize=10) for label, color in color_dict.items()]
    plt.legend(handles=patches)

    plt.show()


if __name__ == "__main__":
    # Debug, este modulo debe ser importado
    consulta = [
        "Similitud Coseno",
        "Errores en codificación MPEG-1",
        "Busqueda eficiente con R-trees",
        "Unigramas, bigramas y trigramas",
    ]
    print(buscar(consulta, 3, sentence_descriptor))
    visualize(load_descriptors(False, text_descriptor))
