import os
import pandas as pd
import busqueda_descr as bus_desc
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import time


def r_prec(ground_values: list[str], search_values: list[str], n=3):
    '''
    dividendo entre los n primeros valores (correctos) y n
    '''
    n_prim = 0
    for value in search_values:
        if value in ground_values:
            n_prim += 1

    return n_prim/n


def mrr(r_precisions: list[int]):
    '''
    promedio de las r_precisions
    '''
    return sum(r_precisions)/len(r_precisions)


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


def calcular_mrr(g_truth, descriptores_locales=None,
                 nombres=None, vectorizer=None, n=10,
                 show_individual=False):
    '''
    Calcula el Mean Reciprocal Rank
    g_truth : matris ground
    descriptores_lcoales : descriptores de los videos del profesor
    nombres: Nombres de los videos del profesor
    vectorizer : vectorizer para calcular descriptores de las consultas
    n : cantidad de valores n que retorno del bus_desc
    '''
    textos_consulta = g_truth['Query'].tolist()

    # Comparamos con busqueda_descr
    result_busc_desc = bus_desc.main(descriptores_locales, nombres,
                                     vectorizer, textos_consulta, n)

    r_precisions = []
    # Algunas métricas
    for key in result_busc_desc.keys():
        resultados = result_busc_desc[key]

        fila_filtrada = g_truth[g_truth['Query'] == key]
        ground_t = fila_filtrada.iloc[0].tolist()

        r_precs = r_prec(ground_t, resultados)
        r_precisions.append(r_precs)

        if show_individual:
            print(f'R-Precission de "{key}": {r_precs:.2f}')

    return mrr(r_precisions)


def main():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    # Cargamos los datos de ground_truth
    g_truth = pd.read_csv(f"{script_dir}/ground_truth.csv",
                          encoding='utf-8', delimiter=';', dtype=str)

    # Calculamos off-line los descriptores de los videos del profesor
    nombres, descriptores_locales, vectorizer = calcular_descriptores_local()

    # Eliminamos los espacios en blanco del dataframe XD (gracias ChatGPT)
    # Iterar a través de todas las columnas del DataFrame
    for columna in g_truth.columns:
        # Verificar si la columna contiene g_truth de tipo cadena (strings)
        if g_truth[columna].dtype == 'object':
            # Eliminar los espacios en blanco de los strings
            g_truth[columna] = g_truth[columna].str.strip()

    mrr_10 = calcular_mrr(g_truth, descriptores_locales,
                          nombres, vectorizer, n=10)
    mrr_20 = calcular_mrr(g_truth, descriptores_locales,
                          nombres, vectorizer, n=20)

    print(f'MRR para n-10 : {mrr_10}')
    print(f'MRR para n-20 : {mrr_20}')


if __name__ == '__main__':
    main()
