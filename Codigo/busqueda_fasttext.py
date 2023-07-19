import scipy.spatial
from unidecode import unidecode
import fasttext
import os
import nltk

# Config
script_dir = os.path.dirname(os.path.abspath(__file__))
transcripts = "{script_dir}/../Videos/Transcripciones/Transcripcion_completa"

# todo: per sentence descriptors and search


def main(texto_consulta: list) -> dict:
    nltk.download('punkt')

    fasttext_model_path = f"{script_dir}/models/cc.es.300.bin"
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
        closest = sorted(distances.items(), key=lambda x: x[1])[:3]
        results[q] = [str(x[0]) for x in closest]

    return results


if __name__ == "__main__":
    consulta = [
        "Similitud Coseno",
        "Errores en codificación MPEG-1",
        "Busqueda eficiente con R-trees",
        "Unigramas, bigramas y trigramas",
    ]
    print(main(consulta))
