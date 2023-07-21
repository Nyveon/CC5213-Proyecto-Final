import numpy as np
import time
import os
import json
import glob
import pickle  # nosec
from sklearn.feature_extraction.text import TfidfVectorizer
from util import normalize


umbral = 0.01
buscador = "tfidf"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
descriptors_file = f"{script_dir}/desc_descr.pkl"

'''
NOTA: Se intentaron metodos como LSA y cdist pero
entregaron peores resultados que la multiplicacion de matrices.
'''


def calcular_descriptores(vectorizer, texto, show=True):
    '''
    Funcion para calcular descriptores
    entrega el tiempo de computo
    '''
    # Calculando descriptores
    t0 = time.time()
    descriptores = vectorizer.transform(texto)
    t1 = time.time()
    if show:
        print("Tiempo descriptores: {:.1f} segs".format(t1-t0))

    return descriptores


def descriptores_textos():
    return calcular_descriptores_local(False)


def descriptores_titulos():
    return calcular_descriptores_local(True)


def calcular_descriptores_local(titulos, show=True):
    '''
    Funcion para calcular descriptores
    de los videos del profesor
    '''
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        sublinear_tf=True,
        norm='l2',
        ngram_range=(1, 1),  # Probar distintos valores
        max_df=1.0,
        # Si una palabra aparece en más que max_df documentos, se ignora
        # Si es float -> porcentaje del total de documentos
        # Si es int   -> cantidad de documentos
        min_df=1   # Si aparece en menos, se ignora, misma idea de int y float
    )

    texts_path = (
        f'{script_dir}/../Videos/Transcripciones/Transcripcion_completa')

    txt_files = glob.glob(texts_path + "/*.txt")

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

    # Calcular el vocabulario
    t0 = time.time()
    vectorizer.fit(textos)
    t1 = time.time()
    print(f'Tiempo para calcular el vocab: {t1-t0}')

    # Calculando descriptores
    t0 = time.time()
    descriptores = vectorizer.transform(textos)
    t1 = time.time()
    if show:
        print("Tiempo descriptores: {:.1f} segs".format(t1-t0))

    return nombres, descriptores, vectorizer


def b_multiplicacion_matrices(nombres, descriptores, textos_consulta,
                              descriptores_consulta, num, show=False):
    '''
    Se busca el más similar mediante multiplicacion
    '''

    t0 = time.time()
    descriptores_f1 = descriptores.toarray()
    descriptores_consulta_f1 = descriptores_consulta.toarray()
    similitudes = np.matmul(descriptores_consulta_f1, descriptores_f1.T)
    t1 = time.time()

    # Mostrar tiempo
    if show:
        print("Tiempo Busqueda Multiplicacion: {:.1f} segs".format(t1-t0))

    values_dict = {}

    indices = np.argsort(-similitudes, axis=1)[:, :num]
    # values = np.take_along_axis(similitudes, indices, axis=1)

    for i in range(len(textos_consulta)):
        values_dict[textos_consulta[i]] = []
        for j in range(num):
            values_dict[textos_consulta[i]].append(nombres[indices[i][j]])

    return values_dict


def comparar_segmentos(vectorizer, json_name: str, query: str) -> float:
    '''
    Se compara cada segmento del video con la consulta,
    Se entrega un valor que sirve para calcular importancia
    '''
    json_data = []
    with open(json_name, encoding='utf-8') as file:
        file.readline()
        file.readline()
        data = json.loads(file.readline())
        for dicts in data:
            json_data.append(dicts['text'])

    descriptores = calcular_descriptores(vectorizer, [query], False)
    descriptores_json = calcular_descriptores(vectorizer, json_data, False)

    descriptores = descriptores.toarray()
    descriptores_json = descriptores_json.toarray()
    similitudes = np.matmul(descriptores_json, descriptores.T)

    count = np.count_nonzero(similitudes > umbral)

    return count/len(json_data)


def obtener_indices_n_mayores(valores, nombres, n=10):
    indices_valores = list(enumerate(valores))
    indices_valores_ordenados = sorted(indices_valores,
                                       key=lambda x: x[1], reverse=True)
    indices_mayores = [indice for indice, _ in indices_valores_ordenados[:n]]

    return [nombres[i] for i in indices_mayores]


def load_descriptors(recalc, f_descriptor):
    file = f"{script_dir}/{buscador}_{f_descriptor.__name__}.pkl"

    if not recalc and os.path.exists(file):
        print("Cargando descriptores pre-calculados...")
        with open(file, "rb") as f:
            nombres, descriptores, vectorizer = pickle.load(f)  # nosec
    else:
        print("Calculando descriptores...")
        nombres, descriptores, vectorizer = f_descriptor()
        with open(file, "wb") as f:
            pickle.dump((nombres, descriptores, vectorizer), f)

    return nombres, descriptores, vectorizer


def buscar(textos_consulta: list, n: int, f_descriptor: callable,
           recalc=False) -> dict:
    """Busca los n videos más similares a cada consulta

    Args:
        texto_consulta (list): lista de queries
        n (int): numero de resultados por query
        f_descriptor (f(str) -> list): Funcion de calculo de descriptores
        recalc (bool, optional): Obligar recalculo de descriptores.

    Returns:
        dict: {query: [video_id1, video_id2, ...]}
    """

    nombres, descriptores, vectorizer = load_descriptors(recalc, f_descriptor)

    # Se calcula la matriz de descriptores para los textos de consulta
    json_path = f'{script_dir}/../Videos/Transcripciones/Transcripcion_json'

    descriptores_consulta = calcular_descriptores(vectorizer, textos_consulta)
    similitudes = b_multiplicacion_matrices(
        nombres, descriptores,
        textos_consulta, descriptores_consulta, 20)

    d_return = {}

    for key, value in similitudes.items():
        l_top = [f'{json_path}/{x}.json' for x in value]
        counts = []
        # Se itera sobre la lista de top-10 para ordenarlas segun importancia
        for video in l_top:
            counts.append(comparar_segmentos(vectorizer, video, key))

        d_return[key] = obtener_indices_n_mayores(counts, value, n)

    return d_return


if __name__ == '__main__':
    textos_consulta = [
        'Similitud Coseno',
    ]

    print(buscar(textos_consulta, 3, calcular_descriptores_local))
