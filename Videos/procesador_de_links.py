import os

# Cambiar el directorio de trabajo al directorio del script
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)

# Procesar los links
archivo_input = "links_sin_procesar.txt"
archivo_output = "links.txt"

with open(archivo_input, "r", encoding="utf-8") as archivo:
    lineas = archivo.readlines()

    videos = {}

    for i in range(len(lineas)):
        linea = lineas[i]
        if linea.startswith('url'):
            nombre = linea.split('-')[0]
            videos[nombre] = lineas[i+1]


with open(archivo_output, "w") as archivo:
    for key, val in videos.items():
        archivo.write(f'{key}\t{val}')
