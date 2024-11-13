from ultralytics import YOLO
import cv2
import math
import os

# Common function to load the model and class names
# Common function to process both image and video detections
detection_sign=""
def process_detection(img,conf_threshold):
    model = YOLO("best.pt")
    classNames = [chr(i) for i in range(65, 91)]  # 'A' to 'Z'
    predicted_sign = None
    results = model(img, stream=True, conf=conf_threshold)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            predicted_sign = classNames[cls]
            
            label = f'{predicted_sign} {conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
    return predicted_sign, img

# Function to handle image detection
def image_detection(image_path):
  # Load model and class names
    img = cv2.imread(image_path)
    predicted_sign, processed_img = process_detection(img,0.45)
    
    # Save processed image
    processed_image_path = os.path.join('static/files', 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_image_path, processed_img)

    return predicted_sign, processed_image_path

# Function to handle video detection
def video_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    global detection_sign
    while True:
        success, img = cap.read()
        if not success:
            break  # Exit loop if no more frames
        detection_sign, processed_img = process_detection(img, 0.45)
        
        # Yield processed frame and detected sign
        yield processed_img, detection_sign
