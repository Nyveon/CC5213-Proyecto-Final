import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple

sys.path.append("..")
import models.busqueda_tfidf as bd     # noqa: E402
import models.busqueda_fasttext as bf  # noqa: E402
import models.busqueda_sbert as bs     # noqa: E402


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
mini_ground_file = f"{script_dir}/mini_ground.txt"


def calc_recall_prec(ground_values: list, s_values: list) -> Tuple[list, list]:
    """Calcula el recall y precision

    Args:
        ground_values (list): Valores correctos
        s_values (list): Valores buscados

    Returns:
        Tuple[list, list]: Tupla de listas de recall y precision
    """
    ind_ = []
    n = len(s_values)
    i = 1

    while len(ind_) < len(ground_values):
        # Si encontramos un valor del ground
        if s_values[i-1] in ground_values:
            ind_.append(i)

        i += 1

        if i > n:
            dummy_l = [-1]*(len(ground_values) - len(ind_))
            ind_ += dummy_l
            break

    recall = []
    preci = []

    for i in ind_:
        if i == -1:
            recall.append(1)
            preci.append(0)
            continue

        recall.append((len(recall) + 1) / len(ground_values))
        preci.append(len(recall) / i)
    return (recall, preci)


def grf_recall_prec(g_truth: pd.DataFrame, n: int, buscador: callable,
                    descriptor: callable, show_individual=False) -> list[list]:
    # TODO: docstring
    textos_consulta = list(g_truth.keys())
    result_busc_desc = buscador(textos_consulta, n, descriptor)

    interpolaciones = []

    for t in textos_consulta:

        ground_t = g_truth[t]
        s_val = result_busc_desc[t]

        recall, precc = calc_recall_prec(ground_t, s_val)
        l_inter = prec_interpolada(precc, recall)
        interpolaciones.append(l_inter)

    promedio_interpolaciones = [0]*11

    for lista in interpolaciones:
        for i in range(11):
            promedio_interpolaciones[i] += lista[i]

    return [valor/len(interpolaciones) for valor in promedio_interpolaciones]


def prec_interpolada(prec: list, recall: list) -> list:
    """Calcula la precision interpolada

    Args:
        prec (list): Lista de precisiones
        recall (list): Lista de recalls

    Returns:
        list: Lista de precisiones interpoladas
    """
    l_rec = []
    l_prec_inter = []

    for i in range(11):
        key = i / 10.0
        l_rec.append(key)

    last_recall = 0

    for i in range(len(l_rec)):
        while l_rec[i] > recall[last_recall]:
            last_recall += 1

        l_prec_inter.append(max(prec[last_recall:]))

    return l_prec_inter


def main() -> None:
    """Test de las funciones de busqueda con el mini ground truth
    """
    with open(mini_ground_file, 'r', encoding='utf-8') as file:
        lineas = file.readlines()

        data_ground = {}

        for linea in lineas:
            linea = linea.strip()
            key = linea.split(';')[0]
            values = linea.split(';')[1:]

            data_ground[key] = [x.strip() for x in values]

    bd_texto = grf_recall_prec(
        data_ground, 10, bd.buscar, bd.text_descriptor)
    bd_titulo = grf_recall_prec(
        data_ground, 10, bd.buscar, bd.title_descriptor)
    bf_texto = grf_recall_prec(
        data_ground, 10, bf.buscar, bf.text_descriptor)
    bf_titulo = grf_recall_prec(
        data_ground, 10, bf.buscar, bf.title_descriptor)
    sb_pm_mpnet = grf_recall_prec(
        data_ground, 10, bs.buscar, bs.pm_mpnet_descriptor)
    sb_distilroberta = grf_recall_prec(
        data_ground, 10, bs.buscar, bs.distilroberta_descriptor)

    ejex = [i * 0.1 for i in range(11)]

    # Crear la gráfica para cada lista
    plt.plot(ejex, bd_texto, marker='o',
             linestyle='-', label='bd_texto n = 25')
    plt.plot(ejex, bd_titulo, marker='o',
             linestyle='-', label='bd_titulo n= 25')
    plt.plot(ejex, bf_texto, marker='o',
             linestyle='-', label='bf_texto n = 25')
    plt.plot(ejex, bf_titulo, marker='o',
             linestyle='-', label='bf_titulo n = 25')
    plt.plot(ejex, sb_pm_mpnet, marker='o',
             linestyle='-', label='sb_pm_mpnet n = 25')
    plt.plot(ejex, sb_distilroberta, marker='o',
             linestyle='-', label='sb_distilroberta n = 25')

    plt.ylim(-0.02, 1)
    plt.xlim(-0.02, 1)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Comparación de sistemas')

    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
