from youtube_transcript_api import YouTubeTranscriptApi
import os
import json

'''
Archivo para transcribir los videos a json files con la API: youtube_transcript

'''

# Cambiar el directorio de trabajo al directorio del script
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)

# Procesar los links
archivo_ = "links.txt"


with open(archivo_, "r", encoding="utf-8") as archivo:
    lineas = archivo.readlines()
    
    videos = {}

    for linea in lineas:
        [nombre, id_] = linea.split('\t')
        id_ = id_.split('watch?v=')[1].replace('\n', '')
        videos[nombre] = id_


ruta_output = f'{script_dir}\Transcripciones'


for nombre, id_ in videos.items():
    print(nombre + '   ' + id_)

    with open(f'{ruta_output}\{nombre}.json', "w") as archivo:

        json_val = YouTubeTranscriptApi.get_transcript(id_, languages=['es'])

        json.dump(json_val, archivo)
