# CC5213-Proyecto-Final

Buscador de videos de catedra NahEri.

## Estructura

```bash
📦CC5213-Proyecto-Final
 ┣ 📂models
 ┃ ┣ 📜busqueda_fasttext.py # Motor de busqueda con fasttext.
 ┃ ┣ 📜busqueda_sbert.py    # Motores de busqueda con SBERT.
 ┃ ┣ 📜busqueda_tfidf.py    # Motor de busqueda con TF-IDF.
 ┃ ┣ 📜_.bin                # Archivos de modelos descargados localmente.
 ┃ ┗ 📜_.pkl                # Archivos de descirptores calculados offline
 ┣ 📂static                 # Javascript y CSS para el front-end
 ┣ 📂templates              # Templates HTML para el front-end
 ┣ 📂tests
 ┃ ┣ 📜_.csv                # Archivos de ground truth para los tests
 ┃ ┣ 📜test_ground.py       # Test con varios ground truths de caso distintos
 ┃ ┗ 📜test_mini_ground.py  # Test visual con un mini-ground truth fijo
 ┣ 📂videos
 ┃ ┣ 📂transcripciones
 ┃ ┃ ┣ 📂transcripcion_completa
 ┃ ┃ ┗ 📂transcripcion_json
 ┃ ┣ 📜procesador_de_links.py   # Pre-procesador de metadatos de videos
 ┃ ┣ 📜titulos.py               # Agregador de titulos de videos
 ┃ ┣ 📜transcribir.py           # Descargador de transcripciones
 ┃ ┣ 📜unir_json.py             # Concatenador de transcripciones
 ┃ ┣ 📜_.txt                    # Listas de videos, URLs y otros meta-datos
 ┃ ┗ 📜_.csv                    # Idem
 ┣ 📜app.py                 # Aplicacion Flask
 ┣ 📜README.md               
 ┣ 📜requirements.txt       # Instalar antes de correr el proyecto
 ┗ 📜util.py                # Configuraciónes y funciones comunes a todos los archivos
```

## Uso

⚠: La primera vez que se corra el proyecto, se debe descargar los modelos de SBERT y FastText. Esto puede tardar un poco.

⚠: La primera vez que se hace una busqueda con un modelo, se debe calcular los descriptores de los videos. Esto puede tardar un poco.

### Instalación

Primero se deben instalar los requisitos:

```bash
pip install -r requirements.txt
```

El proyecto requiere CUDA 11.7. Se puede instalar [aquí](https://developer.nvidia.com/cuda-11-7-0-download-archive)

### Tests de Effectiveness

Para testear la efectividad, se puede ejecutar el archivo `test_ground.py` o `test_mini_ground.py`.
Idealmente se deben correr como modulos.

```bash
python -m tests.test_ground
```

```bash
python -m tests.test_mini_ground
```

### Aplicación Web

Para correr la aplicación web, se debe ejecutar el archivo `app.py`:

```bash
python app.py
```

Una vez lanzado, podrás elegir que buscador usar y hacer consultas.

![Ejemplo Resultados de Busqueda](static/images/image2.png)
