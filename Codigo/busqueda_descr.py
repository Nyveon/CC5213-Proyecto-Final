import numpy as np
import time
import os
import json
import glob
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Config
global umbral
umbral = 0.01
descriptors_file = "descriptors.pkl"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

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
        print("Tiempo descriptores: {:.1f} segs".format(t1-t0), end='\n\n')

    return descriptores


def calcular_descriptores_local(show=True):
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

    # Cambiar el directorio de trabajo al directorio de los textos completos
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    previous_path = os.path.dirname(script_dir)

    texts_path = (
        f'{previous_path}/Videos/Transcripciones/Transcripcion_completa'
    )

    txt_files = glob.glob(texts_path + "/*.txt")

    textos = []
    nombres_completos = []
    nombres = []
    for file_ in txt_files:
        nombres.append(file_.split('\\')[-1].split('.txt')[0])
        with open(file_, 'r', encoding="utf-8") as file:
            nombres_completos.append(file.readline().strip())
            file.readline()
            textos.append(file.readline())

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
        print("Tiempo descriptores: {:.1f} segs".format(t1-t0), end='\n\n')

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
        print("Tiempo Busqueda Multiplicacion: {:.1f} segs".format(t1-t0),
              end='\n\n')

    values_dict = {}

    indices = np.argsort(-similitudes, axis=1)[:, :num]
    # values = np.take_along_axis(similitudes, indices, axis=1)

    for i in range(len(textos_consulta)):
        values_dict[textos_consulta[i]] = []
        for j in range(num):
            values_dict[textos_consulta[i]].append(nombres[indices[i][j]])

    return values_dict


def comparar_segmentos(vectorizer, json_name, query):
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


def buscar(textos_consulta: list, n: int, recalc=False) -> dict:
    """Busca los n videos más similares a cada consulta

    Args:
        texto_consulta (list): lista de queries
        n (int): numero de resultados por query

    Returns:
        dict: {query: [video_id1, video_id2, ...]}
    """

    # Carga los descriptores locales si ya existen o los calcula y guarda.
    if os.path.exists(f"{script_dir}/descriptors.pkl") and not recalc:
        print("Cargando descriptores pre-calculados...")
        with open(f"{script_dir}/descriptors.pkl", "rb") as f:
            nombres, descriptores, vectorizer = pickle.load(f)  # nosec
    else:
        print("Calculando descriptores...")
        nombres, descriptores, vectorizer = calcular_descriptores_local()
        with open(f"{script_dir}/descriptors.pkl", "wb") as f:
            pickle.dump((nombres, descriptores, vectorizer), f)

    return main(descriptores, nombres, vectorizer, textos_consulta, n)


def main(descriptores, nombres, vectorizer, textos_consulta, n):
    # Cambiar el directorio de trabajo al directorio de los textos completos
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    previous_path = os.path.dirname(script_dir)

    json_path = f'{previous_path}/Videos/Transcripciones/Transcripcion_json'
    """
    txt_files = glob.glob(texts_path + "/*.txt")

    textos = []
    nombres_completos = []
    nombres = []
    for file_ in txt_files:
        nombres.append(file_.split('\\')[-1].split('.txt')[0])
        with open(file_, 'r', encoding="utf-8") as file:
            nombres_completos.append(file.readline().strip())
            file.readline()

            textos.append(file.readline())

    # Calcular el vocabulario
    t0 = time.time()
    vectorizer.fit(textos)
    t1 = time.time()
    print(f'Tiempo para calcular el vocab: {t1-t0}')

    ### Calculando descriptores
    descriptores = calcular_descriptores(vectorizer, textos)
    """
    # Se calcula la matriz de descriptores para los textos de consulta
    # (usando el vocabulario)
    descriptores_consulta = calcular_descriptores(vectorizer, textos_consulta)

    similitudes = b_multiplicacion_matrices(
        nombres, descriptores,
        textos_consulta, descriptores_consulta, 20)

    # Diccionario que se retorna, tiene la forma
    # query : [val1, val2, val3]
    d_return = {}

    # Iteramos sobre similitudes para buscar sacar los mas parecidos del top10
    for key, value in similitudes.items():
        l_top = [f'{json_path}/{x}.json' for x in value]
        # Se muestra el top 10.
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

    main(textos_consulta)
