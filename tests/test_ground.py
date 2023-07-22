import os
import pandas as pd
import sys

sys.path.append("..")
import models.busqueda_tfidf as bd     # noqa: E402
import models.busqueda_fasttext as bf  # noqa: E402
import models.busqueda_sbert as bs     # noqa: E402


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


def r_prec(ground_values: list[str], search_values: list[str], n=3) -> float:
    """Calcula el R-Precision

    Args:
        ground_values (list[str]): Valores correctos
        search_values (list[str]): Valores buscados
        n (int, optional): Cantidad de valores correctos. Defaults to 3.

    Returns:
        float: R-Precision
    """
    n_prim = 0
    for value in search_values:
        if value in ground_values:
            n_prim += 1

    return n_prim/n


def mrr(r_precisions: list[int]) -> float:
    """Calcula el Mean Reciprocal Rank

    Args:
        r_precisions (list[int]): Lista de R-Precisions

    Returns:
        float: MRR
    """

    return sum(r_precisions) / len(r_precisions)


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
    result_busc_desc = buscador(textos_consulta, n, descriptor)
    r_precisions = []

    for key in result_busc_desc.keys():
        resultados = result_busc_desc[key]

        fila_filtrada = g_truth[g_truth['Query'] == key]
        ground_t = fila_filtrada.iloc[0].tolist()

        r_precs = r_prec(ground_t, resultados)
        r_precisions.append(r_precs)

        if show_individual:
            print(f'R-Precision de "{key}": {r_precs:.2f}')

    return mrr(r_precisions)


def average_precision(ground_values: list[str],
                      search_values: list[str]) -> float:
    """Calcula el Average Precision

    Args:
        ground_values (list[str]): Valores correctos
        search_values (list[str]): Valores buscados

    Returns:
        float: AP
    """
    scores = 0
    num_hits = 0
    for i, value in enumerate(search_values):
        if value in ground_values:
            num_hits += 1
            scores += num_hits / (i + 1)
    return scores / min(len(ground_values), len(search_values))


def m_ap(average_precisions: list[float]) -> float:
    """Calcula el Mean Average Precision

    Args:
        average_precisions (list[float]): Lista de APs

    Returns:
        float: MAP
    """
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
    result_busc_desc = buscador(textos_consulta, n, descriptor)
    average_precisions = []

    for key in result_busc_desc.keys():
        resultados = result_busc_desc[key]

        fila_filtrada = g_truth[g_truth['Query'] == key]
        ground_t = fila_filtrada.iloc[0].tolist()

        avg_prec = average_precision(ground_t, resultados)
        average_precisions.append(avg_prec)

        if show_individual:
            print(f'Average Precision for "{key}": {avg_prec:.2f}')

    return m_ap(average_precisions)


def bateria_test(g_truth: pd.DataFrame, modulo_buscador: callable,
                 f_descriptor: callable) -> None:
    """Corre una bateria de tests para un buscador y descriptor

    Args:
        g_truth (pd.DataFrame): Matriz ground truth
        modulo_buscador (callable): Modulo de busqueda
        f_descriptor (callable): Funcion de calculo de descriptores
    """
    print((f"\nTesteando buscador {modulo_buscador.__name__}"
           f" usando {f_descriptor.__name__}"))
    mrr10 = calcular_mrr(g_truth, 10, modulo_buscador.buscar, f_descriptor)
    mrr20 = calcular_mrr(g_truth, 20, modulo_buscador.buscar, f_descriptor)
    map3 = calcular_map(g_truth, 3, modulo_buscador.buscar, f_descriptor)
    map20 = calcular_map(g_truth, 20, modulo_buscador.buscar, f_descriptor)
    print(f'MRR para n-10 : {mrr10}')
    print(f'MRR para n-20 : {mrr20}')
    print(f'MAP para n-3 : {map3}')
    print(f'MAP para n-20 : {map20}')


def main() -> None:
    """Corre los tests con los ground truths
    """
    g_truth = pd.read_csv(f"{script_dir}/ground_truth.csv",
                          encoding='utf-8', delimiter=';', dtype=str)

    for columna in g_truth.columns:
        if g_truth[columna].dtype == 'object':
            g_truth[columna] = g_truth[columna].str.strip()

    bateria_test(g_truth, bd, bd.text_descriptor)
    bateria_test(g_truth, bd, bd.title_descriptor)
    bateria_test(g_truth, bf, bf.text_descriptor)
    bateria_test(g_truth, bf, bf.title_descriptor)
    bateria_test(g_truth, bs, bs.pm_mpnet_descriptor)
    bateria_test(g_truth, bs, bs.distilroberta_descriptor)


if __name__ == '__main__':
    main()
