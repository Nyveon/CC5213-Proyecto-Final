import os
import pandas as pd
import busqueda_descr as bd
import busqueda_fasttext as bf
import matplotlib.pyplot as plt
import busqueda_sbert as sb


def calc_recall_prec(ground_values, s_values):
    '''
    Se calcular el recall y prec según una lista de ground truth y una lista
    de valores obtenidos por algunas de las metodologias de busqueda
    '''
    ind_ = []
    n = len(s_values)
    i = 1

    # Mientras nos falte hallar valores del ground
    while len(ind_) < len(ground_values):
        # Si encontramos un valor del ground
        if s_values[i-1] in ground_values:
            ind_.append(i)

        i += 1

        # Agregamos tantos -1 como valores de ground
        # truth no hallamos encontrado en los resultados
        if i > n:
            dummy_l = [-1]*(len(ground_values) - len(ind_))
            ind_ += dummy_l
            break

    # Calculamos el recall y el precision
    recall = []
    preci = []
    
    for i in ind_:
        if i == -1:
            recall.append(1)
            preci.append(0)
            continue

        recall.append((len(recall) +  1) / len(ground_values))
        preci.append(len(recall) / i)
    return recall, preci


'''
(g_truth: pd.DataFrame, n: int, buscador: callable,
                 descriptor: callable, show_individual=False) -> list:
'''

#def recall_and_prec(ground_values : list[str], s_values : list[str], n = 10):
def grf_recall_prec(g_truth: pd.DataFrame, n: int, buscador: callable,
                 descriptor: callable, show_individual=False) -> list[list]: 
    
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

    
        
    

def prec_interpolada(prec, recall):
    '''
    Funcion para calcular la precision interpolada
    '''
    # Los 11 valores de recall
    l_rec = []
    l_prec_inter = [] # para almacenar los valores interpolados
    
    # Crear las llaves en el rango especificado
    for i in range(11):  # Rango del 0 al 1 con incremento de 0.1 (11 elementos)
        key = i / 10.0   # Convertir el índice a su representación decimal
        l_rec.append(key)

    last_recall = 0

    for i in range(len(l_rec)):
        while l_rec[i] > recall[last_recall]:
            last_recall+=1

        l_prec_inter.append(max(prec[last_recall:]))

    return l_prec_inter


def bateria_test(g_truth, modulo_buscador, f_descriptor, n = 10):
    print((f"\nTesteando buscador {modulo_buscador.__name__}"
           f" usando {f_descriptor.__name__} con n = {n}"))
    
    grf_recall_prec(g_truth, n, modulo_buscador.buscar, f_descriptor)


def main():
    ##  Leemos la informacion de mini_ground.txt
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    mini_ground_file = f"{script_dir}/mini_ground.txt"

    with open(mini_ground_file, 'r', encoding='utf-8') as file:
        lineas = file.readlines()

        data_ground= {}

        for linea in lineas:
            l = linea.strip()
            key = l.split(';')[0]
            values = l.split(';')[1:]

            data_ground[key] = [x.strip() for x in values]

    bd_texto = grf_recall_prec(data_ground, 10, bd.buscar, bd.descriptores_textos)
    bd_titulo = grf_recall_prec(data_ground, 10, bd.buscar, bd.descriptores_titulos)
    bf_texto = grf_recall_prec(data_ground, 10, bf.buscar, bf.text_descriptor)
    bf_titulo = grf_recall_prec(data_ground, 10, bf.buscar, bf.title_descriptor)
    sb_pm_mpnet = grf_recall_prec(data_ground, 10, sb.buscar, sb.pm_mpnet_descriptor)
    sb_distilroberta = grf_recall_prec(data_ground, 10, sb.buscar, sb.distilroberta_descriptor)

    
    ejex = [i * 0.1 for i in range(11)]

    # Crear la gráfica para cada lista
    plt.plot(ejex, bd_texto, marker='o', linestyle='-', label='bd_texto n = 25')
    plt.plot(ejex, bd_titulo, marker='o', linestyle='-', label='bd_titulo n= 25')
    plt.plot(ejex, bf_texto, marker='o', linestyle='-', label='bf_texto n = 25')
    plt.plot(ejex, bf_titulo, marker='o', linestyle='-', label='bf_titulo n = 25')
    plt.plot(ejex, sb_pm_mpnet, marker='o', linestyle='-', label='sb_pm_mpnet n = 25')
    plt.plot(ejex, sb_distilroberta, marker='o', linestyle='-', label='sb_distilroberta n = 25')

    # Configurar límites (de 0 a 1)
    plt.ylim(-0.02, 1)
    plt.xlim(-0.02, 1)

    # Configurar etiquetas y título
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Comparación de sistemas')
    
    # Mostrar leyenda
    plt.legend()

    # Mostrar la gráfica
    plt.grid(True)
    plt.show()

main()

