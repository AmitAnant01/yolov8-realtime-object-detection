from ultralytics import YOLO
import cv2
import pyautogui # Required to get screen size

# ----------------- WINDOW POSITIONING SETUP -----------------

# 1. Get the screen size
try:
    screen_width, screen_height = pyautogui.size()
except Exception as e:
    print(f"Warning: Could not get screen size using pyautogui. Window position might be off. Error: {e}")
    # Fallback size
    screen_width, screen_height = 1920, 1080


# Initialize the YOLOv8 Nano model
model = YOLO('yolov8n.pt')

# Get the dictionary of class IDs and names
class_names = model.names

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read the first frame to get the frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read first frame from webcam.")
    exit()

frame_height, frame_width, _ = frame.shape

# Define the window name
WINDOW_NAME = 'YOLOv8 Detection - Press Q to Exit'

# 2. Calculate the center position for the window
# Center X = (Screen Width / 2) - (Frame Width / 2)
# Center Y = (Screen Height / 2) - (Frame Height / 2)
center_x = int((screen_width - frame_width) / 2)
center_y = int((screen_height - frame_height) / 2)

# 3. Create a named window and move it to the calculated position
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(WINDOW_NAME, center_x, center_y)

# ----------------- START DETECTION LOOP -----------------

print("--- Starting YOLOv8 Real-Time Detection ---")

while True:
    # Frame is already read once before the loop, now read the next one
    
    # 2. Perform object detection
    results = model(frame, stream=True)
    
    # Process results from the current frame
    for r in results:
        
        # 3. Draw bounding boxes, labels, and confidence on the frame
        annotated_frame = r.plot()
        
        # 4. Display the annotated frame
        cv2.imshow(WINDOW_NAME, annotated_frame)
        
        # 5. Iterate through detected boxes to print details
        if len(r.boxes) > 0:
            print("\n--- Detections in Current Frame ---")
        
        for box in r.boxes:
            # Extract numerical data
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            box_coordinates = box.xyxy[0].tolist()
            
            # Get the object name using the class ID
            object_name = class_names.get(class_id, "Unknown")
            
            # Print the detailed results
            print(f"Name: {object_name}")
            print(f"  Class ID: {class_id}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Box coordinates (xyxy): {[f'{c:.2f}' for c in box_coordinates]}")

    # 6. Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read the next frame for the next iteration
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

# 7. Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n--- Detection Ended ---")
