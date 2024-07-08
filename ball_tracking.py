import cv2
import numpy as np
import pandas as pd

# Load the video
video_path = 'video.mp4'  
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

quadrant_boundaries = [
    (0, width//2, height//2, height),  # Quadrant 4
    (width//2, width, height//2, height),  # Quadrant 3
    (0, width//2, 0, height//2),  # Quadrant 1
    (width//2, width, 0, height//2)  # Quadrant 2
]
# Color ranges for ball detection
color_ranges = {
    "yellow": ((20, 100, 100), (30, 255, 255)),  
    "green": ((50, 100, 100), (70, 255, 255)),
    "orange": ((10, 100, 100), (20, 255, 255)),
    "white": ((0, 0, 200), (180, 20, 255))
}

# Initialize data storage
events = []
previous_quadrant = None

# Function to detect the quadrant
def detect_quadrant(cx, cy):
    for i, (x1, x2, y1, y2) in enumerate(quadrant_boundaries, start=1):
        if x1 <= cx < x2 and y1 <= cy < y2:
            return i
    return None

print("Processing........... Please wait")
# Process each frame
while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    detected = False
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            
            quadrant = detect_quadrant(cx, cy)
            
            # Check if ball is entering or exiting
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if previous_quadrant != quadrant:
                if previous_quadrant is not None:
                    events.append((current_time, previous_quadrant, color, 'Exit'))
                events.append((current_time, quadrant, color, 'Entry'))
                previous_quadrant = quadrant
            
            # Draw the tracking and text
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 2)
            cv2.putText(frame, f"{color} Entry Q{quadrant} {current_time:.2f}s", 
                        (cx, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            detected = True
            break
    
    if not detected and previous_quadrant is not None:
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        events.append((current_time, previous_quadrant, color, 'Exit'))
        previous_quadrant = None

    # Write the processed frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()

# Save events to a text file
df = pd.DataFrame(events, columns=['Time', 'Quadrant Number', 'Ball Colour', 'Type'])
df.to_csv('events.txt', index=False)

print("Processing complete. Check 'output.avi' and 'events.txt' for results.")



