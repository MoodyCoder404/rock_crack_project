import os, uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from model_module import analyze_image
from gpt_module import generate_explanation

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from werkzeug.utils import secure_filename  # Make sure this import is at the top

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    f = request.files["image"]

    # --- START FIX ---
    # Get the original filename and its extension
    original_filename = secure_filename(f.filename)
    ext = ""
    if '.' in original_filename:
        ext = os.path.splitext(original_filename)[1] # Gets '.jpg', '.png', etc.

    # Create a new unique filename with the *correct* extension
    filename = f"{uuid.uuid4().hex}{ext}"
    # --- END FIX ---

    path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(path)

    try:
        crack_percent, condition = analyze_image(path)  # returns 0–100 and label
    except Exception as e:
        print("Model error:", e)
        return jsonify({"error": f"Model error: {str(e)}"}), 500
    finally:
        # optional cleanup to avoid storing images
        try:
            os.remove(path)
        except Exception:
            pass

    try:
        explanation = generate_explanation(crack_percent, condition)
    except Exception as e:
        explanation = f"(Explanation unavailable: {str(e)})"

    return jsonify({
        "crack_score": float(round(crack_percent, 2)),  # 0–100
        "condition": condition,
        "explanation": explanation
    }), 200


if __name__ == "__main__":
    # Run local dev server
    app.run(host="127.0.0.1", port=5000, debug=True)
