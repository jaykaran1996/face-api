from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/detect-face', methods=['POST'])
def detect_face():

    try:
        # ✅ check file from multipart
        file = request.files.get('image')

        if file is None:
            return jsonify({
                "success": False,
                "message": "No image uploaded"
            }), 400

        # convert file to numpy image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # face detector
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        return jsonify({
            "success": True,
            "face_found": len(faces) > 0,
            "face_count": int(len(faces))
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
