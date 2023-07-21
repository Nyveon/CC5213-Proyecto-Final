import os
import pandas as pd
import busqueda_descr as bd
import busqueda_fasttext as bf


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


def calcular_mrr(g_truth: pd.DataFrame, n: int, buscador: callable,
                 descriptor: callable, show_individual=False) -> list:
    """Calcula el Mean Reciprocal Rank

    Args:
        g_truth (pd.DataFrame): Matriz ground truth
        n (int): Cantidad de resultados a retornar
        buscador (callable): Funcion de busqueda
        descriptor (callable): Funcion de calculo de descriptores
        show_individual (bool, optional): Mostrar MRR de cada query.

    Returns:
        list: Lista de MRRs
    """

    textos_consulta = g_truth['Query'].tolist()

    # Comparamos con busqueda_descr
    result_busc_desc = buscador(textos_consulta, n, descriptor)

    r_precisions = []
    # Algunas métricas
    for key in result_busc_desc.keys():
        resultados = result_busc_desc[key]

        fila_filtrada = g_truth[g_truth['Query'] == key]
        ground_t = fila_filtrada.iloc[0].tolist()

        r_precs = r_prec(ground_t, resultados)
        r_precisions.append(r_precs)

        if show_individual:
            print(f'R-Precision de "{key}": {r_precs:.2f}')

    return mrr(r_precisions)


def average_precision(ground_values: list[str], search_values: list[str]):
    '''
    Calculate the average precision of search_values
    '''
    scores = 0
    num_hits = 0
    for i, value in enumerate(search_values):
        if value in ground_values:
            num_hits += 1
            scores += num_hits / (i + 1)
    return scores / min(len(ground_values), len(search_values))


def m_ap(average_precisions: list[float]):
    '''
    Calculate the mean of the average precisions
    '''
    return sum(average_precisions) / len(average_precisions)


def calcular_map(g_truth: pd.DataFrame, n: int, buscador: callable,
                 descriptor: callable, show_individual=False) -> list:
    """Calculate the Mean Average Precision

    Args:
        g_truth (pd.DataFrame): Ground truth matrix
        n (int): Number of results to return
        buscador (callable): Search function
        descriptor (callable): Descriptor calculation function
        show_individual (bool, optional): Show MAP for each query.

    Returns:
        list: List of MAPs
    """

    textos_consulta = g_truth['Query'].tolist()

    # Compare with busqueda_descr
    result_busc_desc = buscador(textos_consulta, n, descriptor)

    average_precisions = []
    # Some metrics
    for key in result_busc_desc.keys():
        resultados = result_busc_desc[key]

        fila_filtrada = g_truth[g_truth['Query'] == key]
        ground_t = fila_filtrada.iloc[0].tolist()

        avg_prec = average_precision(ground_t, resultados)
        average_precisions.append(avg_prec)

        if show_individual:
            print(f'Average Precision for "{key}": {avg_prec:.2f}')

    return m_ap(average_precisions)


def bateria_test(g_truth, modulo_buscador, f_descriptor):
    print((f"\nTesteando buscador {modulo_buscador.__name__}"
           f" usando {f_descriptor.__name__}"))
    mrr10 = calcular_mrr(g_truth, 10, modulo_buscador.buscar, f_descriptor)
    mrr20 = calcular_mrr(g_truth, 20, modulo_buscador.buscar, f_descriptor)
    map10 = calcular_map(g_truth, 10, modulo_buscador.buscar, f_descriptor)
    map20 = calcular_map(g_truth, 20, modulo_buscador.buscar, f_descriptor)
    print(f'MRR para n-10 : {mrr10}')
    print(f'MRR para n-20 : {mrr20}')
    print(f'MAP para n-10 : {map10}')
    print(f'MAP para n-20 : {map20}')


def main():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    # Cargamos los datos de ground_truth
    g_truth = pd.read_csv(f"{script_dir}/ground_truth.csv",
                          encoding='utf-8', delimiter=';', dtype=str)

    # Eliminamos los espacios en blanco del dataframe XD (gracias ChatGPT)
    # Iterar a través de todas las columnas del DataFrame
    for columna in g_truth.columns:
        # Verificar si la columna contiene g_truth de tipo cadena (strings)
        if g_truth[columna].dtype == 'object':
            # Eliminar los espacios en blanco de los strings
            g_truth[columna] = g_truth[columna].str.strip()

    bateria_test(g_truth, bd, bd.descriptores_textos)
    bateria_test(g_truth, bd, bd.descriptores_titulos)
    bateria_test(g_truth, bf, bf.text_descriptor)
    bateria_test(g_truth, bf, bf.title_descriptor)


if __name__ == '__main__':
    main()
