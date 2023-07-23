import os
import pandas as pd
import sys

from collections import OrderedDict

sys.path.append("..")
import models.busqueda_tfidf as bd      # noqa: E402
import models.busqueda_fasttext as bf   # noqa: E402
import models.busqueda_sbert as bs      # noqa: E402

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


def reciprocal_rank(ground_values: list[str],
                    search_values: list[str]) -> float:
    """Calculates the reciprocal rank

    Args:
        ground_values (list[str]): Correct values
        search_values (list[str]): Searched values

    Returns:
        float: reciprocal rank
    """
    for i, value in enumerate(search_values, 1):
        if value in ground_values:
            return 1 / i
    return 0


def mrr(r_precisions: list[int]) -> float:
    """Calcula el Mean Reciprocal Rank

    Args:
        reciprocal_rank (list[int]): Lista de reciprocal ranks

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
    textos_consulta = list(g_truth.keys())
    result_busc_desc = buscador(textos_consulta, n, descriptor)
    r_precisions = []

    for key in result_busc_desc.keys():
        resultados = result_busc_desc[key]
        ground_t = g_truth[key]

        r_precs = reciprocal_rank(ground_t, resultados)
        r_precisions.append(r_precs)

        if show_individual:
            print(f'MRR "{key}": {r_precs:.2f}')

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


def calcular_map(g_truth: OrderedDict, n: int, buscador: callable,
                 descriptor: callable, show_individual=False) -> list:
    """Calculate the Mean Average Precision

    Args:
        g_truth (OrderedDict): Ground truth {query: [results]}
        n (int): Number of results to return
        buscador (callable): Search function
        descriptor (callable): Descriptor calculation function
        show_individual (bool, optional): Show MAP for each query.

    Returns:
        list: List of MAPs
    """
    textos_consulta = list(g_truth.keys())
    result_busc_desc = buscador(textos_consulta, n, descriptor)
    average_precisions = []

    for key in result_busc_desc.keys():
        resultados = result_busc_desc[key]
        ground_t = g_truth[key]

        avg_prec = average_precision(ground_t, resultados)
        average_precisions.append(avg_prec)

        if show_individual:
            #print(f'gt "{key}": {ground_t}, res {resultados}')
            print(f'AP for "{key}": {avg_prec:.2f}')

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
    map3 = calcular_map(g_truth, 3, modulo_buscador.buscar, f_descriptor)
    print(f'MAP para n-3 : {map3}')
    print(f'MRR para n-10 : {mrr10}')


def load_ground_truth(filename: str) -> OrderedDict:
    """Carga el ground truth desde un archivo

    Args:
        filename (str): Nombre del archivo

    Returns:
        OrderedDict: Ground truth {query: [results]}
    """
    ground_truth = OrderedDict()
    with open(f"{script_dir}/{filename}", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split(',')
            query = line_split[0].strip()
            ground_truth[query] = [x.strip() for x in line_split[1:]]
    return ground_truth


def main() -> None:
    """Corre los tests con los ground truths
    """

    print("-- Caso: Keywords y palabras similares en Titulos --")
    gt = load_ground_truth("gt_titulos.txt")
    bateria_test(gt, bd, bd.title_descriptor)
    bateria_test(gt, bd, bd.title_descriptor_stem)
    # bateria_test(gt, bf, bf.title_descriptor)

    print("-- Caso: Keywords y palabras similares en Texto --")
    gt = load_ground_truth("gt_textos.txt")
    bateria_test(gt, bd, bd.text_descriptor)
    bateria_test(gt, bd, bd.text_descriptor_stem)
    # bateria_test(gt, bf, bf.text_descriptor)
    # bateria_test(gt, bf, bf.sentence_descriptor)
    # bf.model = None
    # bateria_test(gt, bs, bs.pm_mpnet_descriptor)
    # bateria_test(gt, bs, bs.distilroberta_descriptor)

    print("-- Caso: Busqueda semantica --")
    gt = load_ground_truth("gt_semantic.txt")
    bateria_test(gt, bd, bd.text_descriptor_stem)
    # bateria_test(gt, bf, bf.sentence_descriptor)
    # del bd.model
    # bateria_test(gt, bs, bs.pm_mpnet_descriptor)
    # bateria_test(gt, bs, bs.distilroberta_descriptor)


if __name__ == '__main__':
    main()
