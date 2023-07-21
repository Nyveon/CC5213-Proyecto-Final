from flask import Flask, render_template, request
import webview
import os

# Config
transcript_path = "Videos/Transcripciones/Transcripcion_completa"


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
    videos = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = f"{script_dir}/../{transcript_path}"
    for filename in os.listdir(video_dir):
        with open(os.path.join(video_dir, filename),
                  "r", encoding="utf-8") as f:
            title = f.readline()
            url = f.readline()
            video_id = filename.split(".")[0]
            videos.append(Video(title, url, video_id))
    return videos


def main():
    """
    Inicializa la aplicación web
    """

    videos = load_videos()

    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def index():
        query = request.args.get('query', '')
        filtered_list = list(filter(
            lambda x: query.lower() in x.title.lower(), videos))[:3]

        return render_template('index.html', videos=filtered_list)

    webview.create_window('CC5213', app)
    webview.start()


if __name__ == '__main__':
    main()