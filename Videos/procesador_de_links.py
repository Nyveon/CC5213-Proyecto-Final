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
    links = [linea for linea in lineas if linea.startswith('https')]


with open(archivo_output, "w") as archivo:
    for link in links:
        archivo.write(link)
