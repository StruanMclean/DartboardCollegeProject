# Helper function to take photos
import cv2
import time

def set_max_resolution(cap):
    # I had an issue 
    # it kept saying (640, 480) 
    # so i made a check
    resolutions = [
        (3840, 2160),
        (2560, 1440),
        (1920, 1080),
        (1280, 720),
        (640, 480) 
    ]
    
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Check if the resolution was actually set
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width == width and actual_height == height:
            print(f"Camera set to {width}x{height}")
            return width, height
        elif actual_width > 0 and actual_height > 0:
            print(f"Camera supports {actual_width}x{actual_height} (requested {width}x{height})")
            return actual_width, actual_height
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Using default resolution: {width}x{height}")
    return width, height

cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)

if not cap.isOpened() or not cap2.isOpened():
    print("Error: Could not open cameras")
    exit()

print("Setting up camera 1...")
res1 = set_max_resolution(cap)
print("Setting up camera 2...")
res2 = set_max_resolution(cap2)

cap.set(cv2.CAP_PROP_FPS, 30)
cap2.set(cv2.CAP_PROP_FPS, 30)

num = 0

print(f"Camera 1 resolution: {res1[0]}x{res1[1]}")
print(f"Camera 2 resolution: {res2[0]}x{res2[1]}")
print("Starting capture loop...")

while cap.isOpened():
    success1, frame1 = cap.read()
    success2, frame2 = cap2.read()

    if not success1 or not success2:
        print("Could not get frames")
        continue

    # Save preview frames
    cv2.imwrite("frame1.jpg", frame1)
    cv2.imwrite("frame2.jpg", frame2)

    input_from_user = input("Enter c to continue, enter y to save frame: ")
    time.sleep(2)
    
    # Capture fresh frames for saving
    for i in range(5):
        success1, frame1 = cap.read()
        success2, frame2 = cap2.read()
        if success1 and success2:
            cv2.imwrite("frame1.jpg", frame1)
            cv2.imwrite("frame2.jpg", frame2)

    if input_from_user == "c":
        continue
    else:
        cv2.imwrite(f"images/{num}.jpg", frame1)
        cv2.imwrite(f"images/{num + 1}.jpg", frame2)
        print(f"High-resolution image pair {num} saved!")
        num += 2

cap.release()
cap2.release()
cv2.destroyAllWindows()