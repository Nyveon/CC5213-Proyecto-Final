"""
Buscador de videos por similitud de texto
Usando TF-IDF y multiplicacion de matrices.
"""

import numpy as np
import os
import json
import glob
import pickle  # nosec
import sys

from typing import Union, Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append("..")
from util import normalize, transcripts, transcripts_json  # noqa: E402


buscador = "tfidf"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
umbral = 0.01


def text_descriptor() -> Union[list, list, TfidfVectorizer]:
    """Descriptores de transcripciones completas

    Returns:
        Union[list, list, TfidfVectorizer]: Nombres, descriptores y vectorizer
    """
    return calcular_descriptores_local(False)


def title_descriptor() -> Union[list, list, TfidfVectorizer]:
    """Descriptores de titulos de videos

    Returns:
        Union[list, list, TfidfVectorizer]: Nombres, descriptores y vectorizer
    """
    return calcular_descriptores_local(True)


def calcular_descriptores_local(titulos: bool) -> Union[
                                list, list, TfidfVectorizer]:
    """Calcula descriptores TF-IDF (offline)

    Args:
        titulos (bool): Si se usan los titulos o las transcripciones

    Returns:
        Union[list, list, TfidfVectorizer]: Nombres, descriptores y vectorizer
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        sublinear_tf=True,
        norm='l2',
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1
    )

    txt_files = glob.glob(transcripts + "/*.txt")

    transcripciones = []
    nombres_completos = []
    nombres = []
    for file_ in txt_files:
        nombres.append(file_.split('\\')[-1].split('.txt')[0])
        with open(file_, 'r', encoding="utf-8") as file:
            nombres_completos.append(normalize(file.readline()))
            file.readline()
            transcripciones.append(normalize(file.readline()))

    textos = nombres_completos if titulos else transcripciones
    vectorizer.fit(textos)
    descriptores = vectorizer.transform(textos)

    return nombres, descriptores, vectorizer


def b_multiplicacion_matrices(nombres: list, descriptores: list,
                              textos_consulta: list,
                              descriptores_consulta: list,
                              num: int) -> Dict[str, List[str]]:
    """Multiplicacion de matrices para obtener similitudes

    Args:
        nombres (list): Nombres de los videos
        descriptores (list): Descriptores de los videos
        textos_consulta (list): Textos de las consultas
        descriptores_consulta (list): Descriptores de las consultas
        num (int): Numero de resultados por consulta

    Returns:
        Dict[str, List[str]]: {query: [video_id1, video_id2, ...]}
    """

    descriptores_f1 = descriptores.toarray()
    descriptores_consulta_f1 = descriptores_consulta.toarray()
    similitudes = np.matmul(descriptores_consulta_f1, descriptores_f1.T)

    values_dict = {}

    indices = np.argsort(-similitudes, axis=1)[:, :num]

    for i in range(len(textos_consulta)):
        values_dict[textos_consulta[i]] = []
        for j in range(num):
            values_dict[textos_consulta[i]].append(nombres[indices[i][j]])

    return values_dict


def comparar_segmentos(vectorizer: TfidfVectorizer, json_name: str,
                       query: str) -> float:
    """Compara los segmentos de un video con una consulta

    Args:
        vectorizer (TfidfVectorizer): Vectorizador de texto
        json_name (str): Nombre del archivo json
        query (str): Consulta

    Returns:
        float: Porcentaje de segmentos similares
    """
    json_data = []
    with open(json_name, encoding='utf-8') as file:
        file.readline()
        file.readline()
        data = json.loads(file.readline())
        for dicts in data:
            json_data.append(dicts['text'])

    descriptores = vectorizer.transform([query]).toarray()
    descriptores_json = vectorizer.transform(json_data).toarray()

    similitudes = np.matmul(descriptores_json, descriptores.T)

    count = np.count_nonzero(similitudes > umbral)

    return count/len(json_data)


def obtener_indices_n_mayores(valores: list, nombres: list,
                              n: int = 10) -> list:
    """Obtiene los n mayores indices de una lista

    Args:
        valores (list): Lista de valores
        nombres (list):  Lista de nombres
        n (int, optional): Numero de valores a obtener. Default 10

    Returns:
        list: Lista de top n nombres
    """
    indices_valores = list(enumerate(valores))
    indices_valores_ordenados = sorted(indices_valores,
                                       key=lambda x: x[1], reverse=True)
    indices_mayores = [indice for indice, _ in indices_valores_ordenados[:n]]

    return [nombres[i] for i in indices_mayores]


def load_descriptors(recalc: bool, f_descriptor: callable) -> Union[
                     list, list, TfidfVectorizer]:
    """Carga los descriptores pre-calculados o los calcula si no existen

    Args:
        recalc (bool): Obligar recalculo de descriptores.
        f_descriptor (callable): Funcion de calculo de descriptores

    Returns:
        Union[list, list, TfidfVectorizer]: Nombres, descriptores y vectorizer
    """
    file = f"{script_dir}/{buscador}_{f_descriptor.__name__}.pkl"

    if not recalc and os.path.exists(file):
        with open(file, "rb") as f:
            nombres, descriptores, vectorizer = pickle.load(f)  # nosec
    else:
        print("Calculando descriptores por primera vez.")
        nombres, descriptores, vectorizer = f_descriptor()
        with open(file, "wb") as f:
            pickle.dump((nombres, descriptores, vectorizer), f)

    return nombres, descriptores, vectorizer


def buscar(textos_consulta: list, n: int, f_descriptor: callable,
           recalc=False) -> Dict[str, List[str]]:
    """Busca los n videos más similares a cada consulta

    Args:
        texto_consulta (list): lista de queries
        n (int): numero de resultados por query
        f_descriptor (f(str) -> list): Funcion de calculo de descriptores
        recalc (bool, optional): Obligar recalculo de descriptores.

    Returns:
        Dict[str, List[str]]: {query: [video_id1, video_id2, ...]}
    """

    nombres, descriptores, vectorizer = load_descriptors(recalc, f_descriptor)

    descriptores_consulta = vectorizer.transform(textos_consulta)
    similitudes = b_multiplicacion_matrices(
        nombres, descriptores,
        textos_consulta, descriptores_consulta, 20)

    d_return = {}

    for key, value in similitudes.items():
        l_top = [f'{transcripts_json}/{x}.json' for x in value]
        counts = []
        for video in l_top:
            counts.append(comparar_segmentos(vectorizer, video, key))

        d_return[key] = obtener_indices_n_mayores(counts, value, n)

    return d_return


if __name__ == '__main__':
    # Debug, este modulo debe ser importado
    textos_consulta = [
        'Similitud Coseno',
    ]

    print(buscar(textos_consulta, 3, text_descriptor, True))
