import os
import json
import pickle  # nosec
import scipy.spatial
import torch

from sentence_transformers import SentenceTransformer
from util import normalize
from typing import Callable

buscador = "sbert"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
transcripts = f"{script_dir}/../Videos/Transcripciones/Transcripcion_completa"
transcripts_json = f"{script_dir}/../Videos/Transcripciones/Transcripcion_json"

device = torch.device("cuda")
models = {}
model_names = {
    "pm_mpnet_descriptor": "paraphrase-multilingual-mpnet-base-v2",
    "distilroberta_descriptor": "all-distilroberta-v1",
}


def load_model(f_descriptor: callable):
    """Carga el modelo de fasttext si no está cargado
    """
    global models

    if f_descriptor.__name__ not in models:
        models[f_descriptor.__name__] = SentenceTransformer(
            model_names[f_descriptor.__name__],
            device=device)

    return models[f_descriptor.__name__]


def pm_mpnet_descriptor(filename: str) -> list:
    model = load_model(pm_mpnet_descriptor)
    return sentence_descriptor(filename, model)


def distilroberta_descriptor(filename: str) -> list:
    model = load_model(distilroberta_descriptor)
    return sentence_descriptor(filename, model)


def sentence_descriptor(filename: str, model) -> list:
    print(filename)
    descriptors = []
    filename = filename.split(".txt")[0] + ".json"
    with open(os.path.join(transcripts_json, filename),
              "r", encoding="utf-8") as f:
        f.readline()
        f.readline()
        json_text = f.readline()
        json_transcript = json.loads(json_text)
        for fragment in json_transcript:
            descriptor = model.encode(normalize(fragment["text"]))
            descriptors.append(descriptor)

    return descriptors


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
        print("Cargando descriptores pre-calculados...")
        with open(file, "rb") as f:
            return pickle.load(f)  # nosec

    print("Calculando descriptores...")
    vectors = {}

    for filename in os.listdir(transcripts):
        vid = filename.split(".txt")[0]
        vectors[vid] = f_descriptor(filename)

    with open(file, "wb") as f:
        pickle.dump(vectors, f)

    return vectors


def buscar(texto_consulta: list, n: int, f_descriptor: Callable[[str], list],
           recalc=False) -> dict:
    """Busca los n videos más similares a cada query usando fasttext
    Espacio de busqueda: Transcripciones completas

    Args:
        texto_consulta (list): lista de queries
        n (int): numero de resultados por query
        recalc (bool, optional): Obligar recalculo de descriptores
        f_descriptor (f(str) -> list): Funcion de calculo de descriptores

    Returns:
        dict: {query: [video_id1, video_id2, ...]}
    """
    vectors = load_descriptors(recalc, f_descriptor)

    model = load_model(f_descriptor)

    results = {}

    for q in texto_consulta:
        query = normalize(q)
        query_vector = model.encode(query)
        closest = None
        distances = {}

        for video_id, vector_list in vectors.items():
            for vector in vector_list:
                distance = scipy.spatial.distance.cosine(
                    query_vector, vector)
                if video_id not in distances:
                    distances[video_id] = distance
                else:
                    distances[video_id] = min(
                        distances[video_id], distance)

        closest = sorted(distances.items(), key=lambda x: x[1])[:n]
        results[q] = [str(x[0]) for x in closest]

    return results


if __name__ == "__main__":
    consulta = [
        "Similitud Coseno",
        "Errores en codificación MPEG-1",
        "Busqueda eficiente con R-trees",
        "Unigramas, bigramas y trigramas",
    ]
    print(buscar(consulta, 3, pm_mpnet_descriptor))
