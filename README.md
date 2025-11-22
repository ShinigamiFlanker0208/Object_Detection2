Real-Time Object Detection: Contemporary Items, Fruits & Vehicles

📖 Project Overview

This computer vision project utilizes the YOLOv8 (You Only Look Once) architecture to perform real-time object detection via webcam. The system is specifically tuned to identify and classify a curated set of objects relevant to daily life, including humans, common vehicles, fruits, and personal accessories.

Unlike standard detection models that show everything, this project filters the output to focus strictly on specific categories of interest, providing clean visualization with color-coded bounding boxes.

🚀 Features

Instant Detection: leverages the speed of yolov8n (Nano) for smooth real-time performance on standard CPU hardware.

Categorized Visualization:

🔴 Red Boxes: Humans (Person).

🟢 Green Boxes: Vehicles, Fruits, and Daily Items.

Robust Camera Handling: Includes an auto-search feature to detect and connect to the correct video input port (0, 1, or 2).

Confidence Filtering: Automatically ignores low-quality detections (confidence < 40%) to reduce noise.

🎯 Detected Categories

The system is programmed to recognize the following specific objects from the MS COCO dataset:

Category

Objects

Humans

Person

Vehicles

Car, Bicycle, Motorcycle, Bus, Truck

Fruits

Apple, Banana, Orange

Daily Items

Cell Phone, Laptop, Bottle, Cup, Backpack

🛠️ Tech Stack

Python 3.x: The core programming language.

Ultralytics YOLOv8: State-of-the-art object detection model.

OpenCV (cv2): Used for video capture and drawing bounding boxes.

NumPy: Used for efficient array handling.

💻 Installation

Clone or Download this repository to your local machine.

Install Dependencies: Open your terminal/command prompt and run:

pip install opencv-python ultralytics numpy


▶️ How to Run

Connect your webcam to your computer.

Run the main script:

python main.py


Note: The first time you run this, it will automatically download the yolov8n.pt model file (approx. 6MB).

The camera window will open. Point the camera at objects to see detection in action.

Press q to quit the application.

❓ Troubleshooting

"Error: No webcam found": Ensure your camera is not being used by another application (Zoom, Teams, etc.).

Laggy Video: If the video is slow, ensure you are using the yolov8n.pt (Nano) model in the code, not the larger yolov8m or yolov8x versions.

Module Not Found: If you get an error about missing modules, double-check that you ran the pip install command in Step 2.

📚 References

Model: Ultralytics YOLOv8

Dataset: Microsoft COCO (Common Objects in Context)
