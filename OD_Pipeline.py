import cv2
from ultralytics import YOLO

# Load YOLOv8 model
# Consider using 'yolov8s.pt' or 'yolov8m.pt' for better accuracy
model = YOLO('yolov8n.pt')

# Expanded Categories - Now detecting 60+ object types!
# Full COCO dataset has 80 classes - here are the most useful ones
classes_of_interest = {
    # === PEOPLE & ANIMALS ===
    0: "Person",
    14: "Bird", 15: "Cat", 16: "Dog", 17: "Horse",
    18: "Sheep", 19: "Cow", 20: "Elephant", 21: "Bear",
    22: "Zebra", 23: "Giraffe",

    # === VEHICLES ===
    1: "Bicycle", 2: "Car", 3: "Motorcycle", 4: "Airplane",
    5: "Bus", 6: "Train", 7: "Truck", 8: "Boat",

    # === OUTDOOR OBJECTS ===
    9: "Traffic light", 10: "Fire hydrant", 11: "Stop sign",
    12: "Parking meter", 13: "Bench",

    # === ACCESSORIES ===
    24: "Backpack", 25: "Umbrella", 26: "Handbag", 27: "Tie",
    28: "Suitcase", 31: "Skis", 32: "Snowboard",
    33: "Sports ball", 34: "Kite", 35: "Baseball bat",
    36: "Baseball glove", 37: "Skateboard", 38: "Surfboard",

    # === FOOD & KITCHEN ===
    39: "Bottle", 40: "Wine glass", 41: "Cup", 42: "Fork",
    43: "Knife", 44: "Spoon", 45: "Bowl",
    46: "Banana", 47: "Apple", 48: "Sandwich", 49: "Orange",
    50: "Broccoli", 51: "Carrot", 52: "Hot dog", 53: "Pizza",
    54: "Donut", 55: "Cake",

    # === FURNITURE & INDOOR ===
    56: "Chair", 57: "Couch", 58: "Potted plant", 59: "Bed",
    60: "Dining table", 61: "Toilet", 62: "TV", 63: "Laptop",
    64: "Mouse", 65: "Remote", 66: "Keyboard", 67: "Cell phone",

    # === APPLIANCES ===
    68: "Microwave", 69: "Oven", 70: "Toaster", 71: "Sink",
    72: "Refrigerator",

    # === DAILY ITEMS ===
    73: "Book", 74: "Clock", 75: "Vase", 76: "Scissors",
    77: "Teddy bear", 78: "Hair dryer", 79: "Toothbrush"
}

# Open Video Capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("═" * 60)
print("🔍 ENHANCED OBJECT DETECTION")
print(f"📊 Detecting {len(classes_of_interest)} different object types")
print("Press 'q' to quit | Press 'h' to hide/show labels")
print("═" * 60)

show_labels = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run Detection with lower confidence threshold to catch more objects
    results = model(frame, stream=True, verbose=False, conf=0.4)

    detected_objects = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Process if it's a target class
            if cls_id in classes_of_interest:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{classes_of_interest[cls_id]} {conf:.2f}"
                detected_objects.append(classes_of_interest[cls_id])

                # === HUMANS (Yellow) ===
                if cls_id == 0:  # Person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    if show_labels:
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # === ANIMALS (Orange) ===
                elif 14 <= cls_id <= 23:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    if show_labels:
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                # === VEHICLES (Red) ===
                elif cls_id in [1, 2, 3, 4, 5, 6, 7, 8]:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    if show_labels:
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # === FOOD (Magenta) ===
                elif 46 <= cls_id <= 55:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    if show_labels:
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # === ELECTRONICS (Cyan) ===
                elif cls_id in [62, 63, 64, 65, 66, 67]:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    if show_labels:
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # === OTHER OBJECTS (Green) ===
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if show_labels:
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display object count at top
    count_text = f"Objects Detected: {len(detected_objects)}"
    cv2.putText(frame, count_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Enhanced Object Detection', frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        show_labels = not show_labels

cap.release()
cv2.destroyAllWindows()