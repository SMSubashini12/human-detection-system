
from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo11n.pt")

# Analyze the test image
results = model("images/test.jpg.png")

# Check the detected objects
human_found = False

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]

        if label == "person":
            human_found = True
            print("Human detected!")

    # Display the image with detection boxes
    result.show()

if not human_found:
    print("No human detected in the image.")
