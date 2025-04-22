from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (you can use 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
model = YOLO("yolov11n.pt")  # You can change this to any other version

# Load an image
image_path = 'your_image_1.jpg'  # Replace with the path to your image
img = cv2.imread(image_path)

# Run object detection
results = model(image_path)

# Display the image with bounding boxes
results[0].show()

# Optionally save the result
results[0].save(filename="detected_output.jpg")
