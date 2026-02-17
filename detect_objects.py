from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch

# Load your image
image_path = "dog_brains_2500.webp"  # Change this to your image path
image = Image.open(image_path)

# Load model and processor
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# Convert to RGB
image = image.convert("RGB")

# Process image
inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Get predictions
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(
    outputs,
    threshold=0.5,  # Adjust this: lower = more detections, higher = only confident ones
    target_sizes=target_sizes
)[0]

# Print detections
print(f"\n{'='*60}")
print(f"Found {len(results['scores'])} objects in the image:")
print(f"{'='*60}\n")

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )

# Visualize results
image_with_boxes = image.copy()
draw = ImageDraw.Draw(image_with_boxes)

# Try to load a font, fallback to default if not available
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except:
    font = ImageFont.load_default()

# Draw bounding boxes
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x1, y1, x2, y2 = box
    
    # Draw rectangle
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    # Add label with background
    label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}"
    
    # Get text bounding box for background
    bbox = draw.textbbox((x1, y1-20), label_text, font=font)
    draw.rectangle(bbox, fill="red")
    draw.text((x1, y1-20), label_text, fill="white", font=font)

# Save result
output_path = "detected_objects.jpg"
image_with_boxes.save(output_path)
print(f"\n{'='*60}")
print(f"✅ Result saved to: {output_path}")
print(f"{'='*60}\n")

# Display in notebook (if running in Jupyter/Colab)
try:
    from IPython.display import display
    display(image_with_boxes)
except:
    print("💡 Open 'detected_objects.jpg' to see the results!")
