import os
import pandas as pd
import busqueda_descr as bus_desc


def r_prec(ground_values:list[str], search_values:list[str], n = 3):
    '''
    dividendo entre los n primeros valores (correctos) y n
    '''
    n_prim = 0
    for value in search_values:
        if value in ground_values:
            n_prim += 1

    return n_prim/n

def mrr(r_precisions:list[int]):
    '''
    promedio de las r_precisions
    '''
    return sum(r_precisions)/len(r_precisions)



def comparar():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    # Cargamos los datos de ground_truth
    g_truth = pd.read_csv(f"{script_dir}\ground_truth.csv", encoding='utf-8', delimiter=';', 
                          dtype=str)

    textos_consulta = g_truth['Query'].tolist()

    # Comparamos con busqueda_descr
    result_busc_desc = bus_desc.main(textos_consulta, 10)
    



comparar()