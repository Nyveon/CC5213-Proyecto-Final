import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import glob
from scipy.spatial import distance
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import svd


def calcular_descriptores(vectorizer, texto):
    '''
    Funcion para calcular descriptores
    entrega el tiempo de computo
    '''
    ### Calculando descriptores
    t0 = time.time()
    descriptores = vectorizer.transform(texto)
    t1 = time.time()
    print("Tiempo descriptores: {:.1f} segs".format(t1-t0), end = '\n\n')

    return descriptores

def b_multiplicacion_matrices(nombres, descriptores, textos_consulta, descriptores_consulta):
    # Se busca el más similar
    print('Buscando similares mediante multiplicación de matrices')

    t0 = time.time()
    descriptores_f1= descriptores.toarray()
    descriptores_consulta_f1 = descriptores_consulta.toarray()
    similitudes = np.matmul(descriptores_consulta_f1, descriptores_f1.T)
    t1 = time.time()

    mayor = np.amax(similitudes, axis=1)
    posicion_mayor = np.argmax(similitudes, axis=1)

    for i in range(len(textos_consulta)):
        print(f"{textos_consulta[i]} -- {nombres[posicion_mayor[i]]} -- {mayor[i]}")
    
    print("Tiempo Busqueda Multiplicacion: {:.1f} segs".format(t1-t0), end='\n\n')

def b_cdist(nombres, descriptores, textos_consulta, descriptores_consulta):

    print('Buscando similares mediante Cdist')
    t0 = time.time()
    descriptores_f2 = descriptores.toarray()
    descriptores_consulta_f2 = descriptores_consulta.toarray()
    similitudes = distance.cdist(descriptores_consulta_f2, descriptores_f2, metric='cosine') # Se pueden probar otras distancias
    t1 = time.time()

    mayor = np.amin(similitudes, axis=1)
    posicion_mayor = np.argmin(similitudes, axis=1)

    for i in range(len(textos_consulta)):
        print(f"{textos_consulta[i]} -- {nombres[posicion_mayor[i]]} -- {mayor[i]}")


    print("Tiempo Busqueda Cdist: {:.1f} segs".format(t1-t0), end='\n\n')

### FUNCIONA MAAAL!!!
def b_LSA(nombres, descriptores, textos_consulta, descriptores_consulta):
    print('Buscando similares mediante LSA')

    t0 = time.time()
    transformer_tsvd = TruncatedSVD(n_components=100, n_iter=10, random_state=1)
    transformer_tsvd.fit(descriptores)
    transformer_tsvd.fit(descriptores_consulta)
    t1 = time.time()
    print("tiempo ajustar la transformacion: {:.1f} segs".format(t1-t0))

    descriptores_tsvd = transformer_tsvd.transform(descriptores)
    descriptores_consulta_tsvd = transformer_tsvd.transform(descriptores_consulta)

    t0 = time.time()
    similitudes_tsvd = np.matmul(descriptores_tsvd, descriptores_consulta_tsvd.T)
    t1 = time.time()
    print("tiempo comparacion todos contra todos: {:.1f} segs".format(t1-t0))
    
    mayor = np.amax(similitudes_tsvd, axis=1)
    posicion_mayor = np.argmax(similitudes_tsvd, axis=1)

    for i in range(len(textos_consulta)):
        print(f"{textos_consulta[i]} -- {nombres[posicion_mayor[i]]} -- {mayor[i]}")



def main(vectorizer, textos_consulta):

    # Cambiar el directorio de trabajo al directorio de los textos completos
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    previous_path = os.path.dirname(script_dir)

    texts_path = f'{previous_path}\Videos\Transcripciones\Transcripcion_completa'

    txt_files = glob.glob(texts_path + "/*.txt")

    textos = []
    nombres = []
    for file_ in txt_files:
        nombres.append(file_.split('\\')[-1])

        with open(file_, 'r', encoding="utf-8") as file:
            textos.append(file.read())

    # Calcular el vocabulario
    t0 = time.time()
    vectorizer.fit(textos)
    t1 = time.time()
    print(f'Tiempo para calcular el vocab: {t1-t0}')

    ### Calculando descriptores
    descriptores = calcular_descriptores(vectorizer, textos)

    # Se calcula la matriz de descriptores para los textos de consulta (usando el vocabulario)
    descriptores_consulta = calcular_descriptores(vectorizer, textos_consulta)

    b_multiplicacion_matrices(nombres, descriptores, textos_consulta, descriptores_consulta)

    b_cdist(nombres, descriptores, textos_consulta, descriptores_consulta)

    b_LSA(nombres, descriptores, textos_consulta, descriptores_consulta)


if __name__ == '__main__':
    vectorizer = TfidfVectorizer(
    lowercase = True,
    strip_accents = 'unicode',
    sublinear_tf = True,
    norm = 'l2',
    ngram_range = (1,1), # Probar distintos valores
    max_df = 1.0, # Si una palabra aparece en más que max_df documentos, se ignora
                  #  Si es float -> porcentaje del total de documentos
                  #  Si es int   -> cantidad de documentos
    min_df = 1   # Si aparece en menos, se ignora, misma idea de int y float
    )
    
    textos_consulta = [
    'Similitud Coseno',
    'Machine Learning',
    'Descriptores de videos',
    ]


    main(vectorizer, textos_consulta)