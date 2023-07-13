import webview
from flask import Flask, render_template, request


class Video:
    """Representación abstracta de un video de YouTube
    """
    def __init__(self, title, url):
        self.title = title
        self.url = url

    def getEmbedUrl(self):
        return self.url.replace("watch?v=", "embed/")


videos = [
    Video(
        "1.1-Introducción a la Recuperación de Información Multimedia",
        "https://www.youtube.com/watch?v=uGAuiAgnTJ0"
    ),
    Video(
        "1.2-Introducción a OpenCV",
        "https://www.youtube.com/watch?v=499Uq5_UeE4"
    ),
    Video(
        "1.3-Procesamiento de imágenes. Parte 1 de 2. Operadores punto a punto", # noqa
        "https://www.youtube.com/watch?v=fGCsvc1LwdE"
    )
]


def main():
    """Inicializa la aplicación web
    """
    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def index():
        query = request.args.get('query', '')
        filtered_list = list(filter(
            lambda x: query.lower() in x.title.lower(), videos))

        return render_template('index.html', videos=filtered_list)

    webview.create_window('CC5213', app)
    webview.start()


if __name__ == '__main__':
    main()
