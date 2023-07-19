import scipy.spatial
from unidecode import unidecode
import fasttext
import fasttext.util
import os
import nltk

# Config
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
transcripts = f"{script_dir}/../Videos/Transcripciones/Transcripcion_completa"


def main(texto_consulta: list, n: int) -> dict:
    """Busca los n videos más similares a cada query usando fasttext

    Args:
        texto_consulta (list): lista de queries
        n (int): numero de resultados por query

    Returns:
        dict: {query: [video_id1, video_id2, ...]}
    """
    nltk.download('punkt')

    fasttext.util.download_model("es", if_exists="ignore")
    fasttext_model_path = "cc.es.300.bin"
    model = fasttext.load_model(os.path.join(script_dir, fasttext_model_path))
    print("palabras =", len(model.words))
    print("dimensión =", model.get_dimension())

    vectors = {}

    results = {}

    for filename in os.listdir(transcripts):
        print(filename)
        with open(os.path.join(transcripts, filename),
                  "r", encoding="utf-8") as f:
            f.readline()
            f.readline()
            text = unidecode(f.read()).lower()
            video_id = filename.split(".txt")[0]
            vectors[video_id] = model.get_sentence_vector(text)

    for q in texto_consulta:
        query = unidecode(q.lower())
        query_vector = model.get_sentence_vector(query)
        closest = None
        distances = {}
        for video_id, vector in vectors.items():
            distance = scipy.spatial.distance.cosine(query_vector, vector)
            distances[video_id] = distance
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
    print(main(consulta, 3))
