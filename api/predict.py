from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# تحميل النموذج
model = YOLO('final_model.pt')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        results = model(image, conf=0.35)
        
        boxes = results[0].boxes
        leishmania = sum(1 for b in boxes if int(b.cls[0]) == 0)
        macrophage = sum(1 for b in boxes if int(b.cls[0]) == 1)
        
        annotated = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated)
        result_image = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'leishmania': leishmania,
            'macrophage': macrophage,
            'image': result_image
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
