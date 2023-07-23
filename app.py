import models.busqueda_tfidf as tfidf
import models.busqueda_fasttext as fasttext
import models.busqueda_sbert as sbert
import webview
import os

from flask import Flask, render_template, request
from util import transcripts


class Video:
    """
    Representación abstracta de un video de YouTube
    """
    def __init__(self, title, url, video_id):
        self.video_id = video_id
        self.title = title
        self.url = url
        self.youtube_id = self.url.split("watch?v=")[1]
        self.embed_url = f"https://www.youtube.com/embed/{self.youtube_id}"


def load_videos():
    """
    Carga todos los videos a objetos para poder ser usados en la aplicación
    """
    videos = {}

    for filename in os.listdir(transcripts):
        with open(os.path.join(transcripts, filename),
                  "r", encoding="utf-8") as f:
            title = f.readline()
            url = f.readline()
            video_id = ".".join(filename.split(".")[:-1])
            videos[video_id] = Video(title, url, video_id)
    return videos


buscadores = {
    "TF-IDF Títulos": (
        tfidf, tfidf.title_descriptor),
    "TF-IDF Títulos Stemmizados": (
        tfidf, tfidf.title_descriptor_stem),
    "TF-IDF Títulos Lemmatizados": (
        tfidf, tfidf.title_descriptor_lem),
    "TF-IDF Textos": (
        tfidf, tfidf.text_descriptor),
    "TF-IDF Textos Stemmizados": (
        tfidf, tfidf.text_descriptor_stem),
    "Fasttext Textos": (
        fasttext, fasttext.text_descriptor),
    "Fasttext Fragmentos": (
        fasttext, fasttext.sentence_descriptor),
    "Fasttext Títulos": (
        fasttext, fasttext.title_descriptor),
    "S-BERT MPNet": (
        sbert, sbert.pm_mpnet_descriptor),
    "S-BERT DistilRoberta": (
        sbert, sbert.distilroberta_descriptor),
}


def main():
    """
    Inicializa la aplicación web
    """
    videos = load_videos()
    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def index():
        buscador = request.args.get('buscador', None)
        if buscador is not None:
            modulo_buscador, funcion_buscador = buscadores[buscador]
            print(buscador)

        filtered_list = []
        query = request.args.get('query', None)
        if query is not None:
            search_result = modulo_buscador.buscar(
                [query], 3, funcion_buscador)
            print(search_result)

            for result in search_result[query]:
                filtered_list.append(videos[result])

        return render_template('index.html', videos=filtered_list,
                               query=query, buscadores=buscadores.keys(),
                               selected_buscador=buscador)

    webview.create_window('CC5213', app)
    webview.start()


if __name__ == '__main__':
    main()
