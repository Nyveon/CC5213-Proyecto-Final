import os

'''
Archivo para procesar los links de los videos del curso
'''

# Cambiar el directorio de trabajo al directorio del script
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)

# Procesar los links
archivo_input = "links_sin_procesar.txt"
archivo_output = "videos.csv"
ids = {}

with open(archivo_input, "r", encoding="utf-8") as archivo:
    lineas = archivo.readlines()

    videos = []

    for i in range(len(lineas)):
        linea = lineas[i]
        if linea.startswith('url'):
            nombre = linea.split('url')[1].strip()

            # Contador para hacer unicos los ids de vídeo
            video_id = nombre.split('-')[0]
            if video_id in ids:
                ids[video_id] += 1
                video_id = f"{video_id}.{ids[video_id]}"
            else:
                ids[video_id] = 1

            url = lineas[i+1].strip()
            videos.append((nombre, video_id, url))

# Guardar CSV
with open(archivo_output, "w", encoding="utf-8") as archivo:
    archivo.write("nombre\tid\turl\n")
    for nombre, video_id, url in videos:
        archivo.write(f"{nombre}\t{video_id}\t{url}\n")
