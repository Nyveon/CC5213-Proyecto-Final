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
# CSV tab separated: "nombre id url"
archivo_ = "videos.csv"

videos = []

with open(archivo_, "r", encoding="utf-8") as archivo:
    lineas = archivo.readlines()[1:]

    for linea in lineas:
        [nombre, video_id, url] = linea.split('\t')
        youtube_id = url.split('watch?v=')[1].replace('\n', '')
        videos.append((nombre, video_id, youtube_id, url))


ruta_output = f'{script_dir}/Transcripciones'

for nombre, video_id, youtube_id, url in videos:
    print(nombre + '   ' + video_id)

    with open(f'{ruta_output}/{video_id}.json',
              "w", encoding="utf-8") as archivo:
        archivo.write(f"{nombre}\n{url}")
        json_val = YouTubeTranscriptApi.get_transcript(youtube_id,
                                                       languages=['es'])
        json.dump(json_val, archivo)
