import scipy.spatial
import fasttext
import fasttext.util
import os
import pickle  # nosec
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from typing import Callable
from util import normalize

# Config
buscador = "fasttext"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
transcripts = f"{script_dir}/../Videos/Transcripciones/Transcripcion_completa"
descriptors_file = f"{script_dir}/fasttext.pkl"

model = None


def load_model():
    """Carga el modelo de fasttext si no está cargado
    """
    global model

    if model is None:
        fasttext.util.download_model("es", if_exists="ignore")
        fasttext_model_path = "cc.es.300.bin"
        model = fasttext.load_model(
            os.path.join(script_dir, fasttext_model_path))


def text_descriptor(filename: str) -> list:
    """Función Descriptor. Extrae descriptores la transcripción.

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
    load_model()
    vectors = load_descriptors(recalc, f_descriptor)
    # visualize(vectors)

    results = {}

    for q in texto_consulta:
        query = normalize(q)
        query_vector = model.get_sentence_vector(query)
        closest = None
        distances = {}
        for video_id, vector in vectors.items():
            distance = scipy.spatial.distance.cosine(query_vector, vector)
            distances[video_id] = distance
        closest = sorted(distances.items(), key=lambda x: x[1])[:n]
        results[q] = [str(x[0]) for x in closest]

    return results


def visualize(vectors):
    """Visualize the vectors in 2D space.

    Args:
        vectors (dict): The vectors to visualize.
    """
    # Convert the vectors to a 2D numpy array
    vectors_list = np.array(list(vectors.values()))
    video_ids = list(vectors.keys())

    # Extract the first two characters from each video ID and get unique ones
    labels = [vid[:2] for vid in video_ids]
    unique_labels = list(set(labels))

    # Map each unique label to a color
    colormap = plt.cm.get_cmap('tab20', len(unique_labels))
    color_dict = {label: colormap(i) for i, label in enumerate(unique_labels)}

    # Use t-SNE to reduce the vectors to two dimensions
    tsne = TSNE(n_components=2, random_state=0)
    vectors_2d = tsne.fit_transform(vectors_list)

    # Create a scatter plot of the 2D vectors
    plt.figure(figsize=(10, 10))

    for i, video_id in enumerate(video_ids):
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1],
                    color=color_dict[labels[i]])
        plt.annotate(video_id, (vectors_2d[i, 0], vectors_2d[i, 1]))

    # Create a legend
    patches = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color,
                          markersize=10) for label, color in color_dict.items()]
    plt.legend(handles=patches)

    plt.show()


if __name__ == "__main__":
    consulta = [
        "Similitud Coseno",
        "Errores en codificación MPEG-1",
        "Busqueda eficiente con R-trees",
        "Unigramas, bigramas y trigramas",
    ]
    print(buscar(consulta, 3, text_descriptor))
