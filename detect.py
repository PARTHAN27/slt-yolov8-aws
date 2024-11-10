from ultralytics import YOLO
import cv2
import math
from gtts import gTTS
import playsound
import os

def play_detection_sound(letter):
    global last_spoken_letter
    if letter != last_spoken_letter:  # Speak only if the letter has changed
        # Generate audio using gTTS
        tts = gTTS(text=letter, lang='en')
        
        # Create a custom temporary directory
        temp_dir = os.path.join(os.path.expanduser("~"), "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)  # Create the directory if it doesn't exist
        
        temp_file_path = os.path.join(temp_dir, f"{letter}.mp3")
        tts.save(temp_file_path)
        playsound.playsound(temp_file_path)
        last_spoken_letter = letter  # Update the last spoken letter

def video_detection(path_x):
    global last_spoken_letter
    last_spoken_letter = None  # Initialize the last spoken letter

    video_capture = path_x
    # Create a Webcam Object
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Uncomment if you want to save output to a file
    # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    model = YOLO("best.pt")
    classNames = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    ]

    while True:
        success, img = cap.read()
        if not success:
            break  # Exit if the video capture fails

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                
                # Play sound for the detected letter
                play_detection_sound(class_name)

                label = f'{class_name} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Display the image
        #cv2.imshow("Image", img)
                yield img

        # Uncomment if you want to save output to a file
        # out.write(img)

        # Exit on 'q' key press
        

    # Uncomment if you want to save output to a file
    # out.release()
    #cap.release()
    cv2.destroyAllWindows()

# Call the function with your video path
