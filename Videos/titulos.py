import os
import glob

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
json_dir = f'{script_dir}/Transcripciones/Transcripcion_json'
os.chdir(json_dir)
json_files = glob.glob("*.json")

nombres = []
for file_name in json_files:
    with open(f'{json_dir}/{file_name}',  encoding="utf-8") as file:
        nombres.append(file.readline())

with open(f'{script_dir}/titulos.txt', "w",  encoding="utf-8") as file:
    for nombre in nombres:
        file.write(nombre)
