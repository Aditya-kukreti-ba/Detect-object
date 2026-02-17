from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import io
import base64

app = Flask(__name__)
CORS(app)  # Allow requests from your React app

# Load model once at startup (faster for multiple requests)
print("Loading YOLOS model...")
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
print("✅ Model loaded successfully!")

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        # Get image from request
        data = request.json
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB (same as your code)
        image = image.convert("RGB")
        
        # Process image (same as your code)
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        # Get predictions (same as your code)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(
            outputs,
            threshold=0.5,  # You can make this adjustable via query params
            target_sizes=target_sizes
        )[0]
        
        # Format response for frontend
        detections = []
        width, height = image.size
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            
            detections.append({
                "object": model.config.id2label[label.item()],
                "confidence": int(score.item() * 100),
                "bbox": [
                    round(x1/width*100, 1),  # Convert to percentages
                    round(y1/height*100, 1),
                    round(x2/width*100, 1),
                    round(y2/height*100, 1)
                ]
            })
        
        print(f"✅ Detected {len(detections)} objects")
        return jsonify(detections)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "yolos-tiny"})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 YOLOS Object Detection API")
    print("="*60)
    print("Server running on: http://localhost:5000")
    print("Try: http://localhost:5000/health")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
