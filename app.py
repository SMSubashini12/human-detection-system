
from flask import Flask, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs("uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

model = YOLO("yolo11n.pt")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    photo = request.files.get("photo")

    if not photo or photo.filename == "":
        return "Please select a photo.", 400

    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    extension = os.path.splitext(photo.filename)[1].lower()

    if extension not in allowed_extensions:
        return "Please upload a JPG, JPEG, PNG, or WEBP image.", 400

    filename = secure_filename(photo.filename)
    upload_path = os.path.join("uploads", filename)
    photo.save(upload_path)

    results = model(upload_path)
    result = results[0]

    human_found = any(
        model.names[int(box.cls[0])] == "person"
        for box in result.boxes
    )

    result_filename = "result_" + filename
    result_path = os.path.join("static", "results", result_filename)
    result.save(filename=result_path)

    return render_template(
        "index.html",
        human_found=human_found,
        result_image="/" + result_path.replace("\\", "/")
    )


if __name__ == "__main__":
    app.run(debug=True)
