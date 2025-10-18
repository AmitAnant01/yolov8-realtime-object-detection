# IMPORTING THE REQUIRED LIBRARIES FOR THIS PROJECT
from ultralytics import YOLO
import cv2
import pyautogui 

try:
    screen_width, screen_height = pyautogui.size()
except Exception as e:
    print(f"Warning: Could not get screen size using pyautogui. Window position might be off. Error: {e}")
    screen_width, screen_height = 1920, 1080


# Initialize the YOLOv8 Nano model
model = YOLO('yolov8n.pt')


class_names = model.names

# This will start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read first frame from webcam.")
    exit()

frame_height, frame_width, _ = frame.shape

# Define the window name
WINDOW_NAME = 'YOLOv8 Detection - Press Q to Exit' # During the Detecting the Object if we want to quite so press Q.

center_x = int((screen_width - frame_width) / 2)
center_y = int((screen_height - frame_height) / 2)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(WINDOW_NAME, center_x, center_y)

print("--- Starting YOLOv8 Real-Time Detection ---")

while True:
    results = model(frame, stream=True)
    for r in results:
        annotated_frame = r.plot()
        cv2.imshow(WINDOW_NAME, annotated_frame)
        if len(r.boxes) > 0:
            print("\n--- Detections in Current Frame ---")
        
        for box in r.boxes:
            # Extract numerical data
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            box_coordinates = box.xyxy[0].tolist()
            object_name = class_names.get(class_id, "Unknown")
            
            print(f"Name: {object_name}")
            print(f"  Class ID: {class_id}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Box coordinates (xyxy): {[f'{c:.2f}' for c in box_coordinates]}")
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break


cap.release()
cv2.destroyAllWindows()
print("\n Detection has been ended!")

