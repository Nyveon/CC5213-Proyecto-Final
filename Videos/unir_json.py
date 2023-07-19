import os
import glob
import json

'''
Para hacer merge de todos los subtitulos de los json files
'''

# Cambiar el directorio de trabajo al directorio de los json
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
json_dir = f'{script_dir}\Transcripciones\Transcripcion_json'
os.chdir(json_dir)

save_dir = f'{script_dir}\Transcripciones\Transcripcion_completa'

json_files = glob.glob("*.json")

for file_name in json_files:
    # Leemos el archivo    
    # print(file_name)
    with open(f'{json_dir}\{file_name}') as file:
        # Unimos sus keys
        data = json.load(file)

        # Unimos los textos
        text = ''
        for dict_ in data: text += f" {dict_['text']}"

        # Guardamos en un .txt
    file_ = file_name[:-5] + '.txt'
    with open(f'{save_dir}\{file_}', "w",  encoding="utf-8") as file:
        file.write(text)

        