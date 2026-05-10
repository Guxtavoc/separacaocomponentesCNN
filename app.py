from flask import Flask, render_template, request
from analise import *

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analisar", methods=["POST"])
def analisar():

    imagem = request.files["imagem"]

    caminho = "static/uploads/" + "imagem.jpg"

    imagem.save(caminho)

    resultado = analise(caminho)

    return render_template ("resultado.html",resultado=resultado)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)