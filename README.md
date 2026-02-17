# YOLOS Object Detection Web App

A complete web application for real-time object detection using the YOLOS-Tiny model from HuggingFace.

## 🎯 What You Get

- **Python Script** (`detect_objects.py`) - Test the model locally with any image
- **Flask Backend** (`flask_backend.py`) - REST API for object detection
- **React Frontend** (`yolos-detector-app.jsx`) - Beautiful web interface
- Full visualization with bounding boxes and confidence scores

## 🚀 Quick Start

### Option 1: Test Locally (Python Script)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
python detect_objects.py
```

Replace `"dog_brains_2500.webp"` in the script with your image path.

### Option 2: Full Web App

#### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 2: Start the Backend

```bash
python flask_backend.py
```

You should see:
```
🚀 YOLOS Object Detection API
Server running on: http://localhost:5000
```

#### Step 3: Open the Frontend

1. Go to claude.ai
2. Upload `yolos-detector-app.jsx`
3. Click to open the artifact
4. Upload an image and click "Detect Objects"!

## 📁 Files Explained

### `detect_objects.py`
Standalone script to test object detection on a single image. Saves results to `detected_objects.jpg`.

### `flask_backend.py`
REST API server that:
- Loads the YOLOS model once on startup
- Accepts base64-encoded images via POST
- Returns detected objects with bounding boxes
- Runs on http://localhost:5000

### `yolos-detector-app.jsx`
React web interface with:
- Drag-and-drop image upload
- Real-time object detection
- Visual bounding boxes
- Confidence scores
- Backend status indicator

### `requirements.txt`
All Python dependencies needed for the project.

## 🎨 Customization

### Adjust Detection Threshold

In `flask_backend.py`, change the threshold:

```python
results = image_processor.post_process_object_detection(
    outputs,
    threshold=0.3,  # Lower = more detections, Higher = only confident ones
    target_sizes=target_sizes
)[0]
```

- `0.3` - Detect more objects (may include false positives)
- `0.5` - Balanced (default)
- `0.7` - Only very confident detections

### Use a Different Model

Replace `'hustvl/yolos-tiny'` with:
- `'hustvl/yolos-small'` - Better accuracy, slower
- `'hustvl/yolos-base'` - Even better, but requires more memory

## 🌐 Deploy to Production

### Backend Options:

**Render (Free Tier)**
1. Push your code to GitHub
2. Go to render.com → New → Web Service
3. Connect your repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `python flask_backend.py`

**Railway**
1. Push to GitHub
2. Go to railway.app → New Project
3. Select your repo
4. Railway auto-detects Python and installs dependencies

**Fly.io**
```bash
fly launch
fly deploy
```

### Frontend:

The React app works directly in claude.ai, or you can:

**Deploy to Vercel/Netlify:**
1. Convert to a full React app with Create React App
2. Update the API URL to your deployed backend
3. Deploy!

## 🔧 Troubleshooting

### Backend won't start
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 5000 is available: `lsof -i :5000`

### CORS errors
- Make sure `flask-cors` is installed
- Backend must be running when you use the frontend

### "Backend Offline" message
- Start the Flask server: `python flask_backend.py`
- Check that it's running on http://localhost:5000

### No objects detected
- Lower the threshold in `flask_backend.py`
- Try different images (model trained on COCO dataset)
- Make sure image is clear and well-lit

## 📊 What Can It Detect?

The YOLOS model is trained on the COCO dataset and can detect 80+ object categories:
- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle, airplane, train
- **Animals**: dog, cat, bird, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Kitchen**: bottle, wine glass, cup, fork, knife, spoon, bowl
- **Furniture**: chair, couch, bed, dining table, toilet
- **Electronics**: tv, laptop, mouse, keyboard, cell phone
- **Sports**: sports ball, baseball bat, skateboard, surfboard, tennis racket
- And many more!

## 🎓 Learning Resources

- [YOLOS Paper](https://arxiv.org/abs/2106.00666)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Flask Documentation](https://flask.palletsprojects.com/)

## 💡 Next Steps

1. Add batch processing for multiple images
2. Implement video detection (frame-by-frame)
3. Add custom model fine-tuning
4. Create a mobile app version
5. Add object tracking across frames

## ❓ Need Help?

Common issues:
- **Model too slow?** → Use GPU: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- **Out of memory?** → Use a smaller image or increase system RAM
- **Wrong detections?** → Adjust threshold or try a larger model

Enjoy detecting objects! 🎯
