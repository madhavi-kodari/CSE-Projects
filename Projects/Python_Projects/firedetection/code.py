import cv2
import numpy as np
import playsound
import threading

# Global variables to manage alarm and fire detection state
Alarm_Status = False  # Flag to check if the alarm is playing

def play_alarm_sound_function():
    """
    Function to play the alarm sound in a loop using threading.
    This ensures the alarm can play without blocking the main program.
    """
    while Alarm_Status:  # Play sound only if the alarm status is True
        playsound.playsound("audio.sound.mp3", True)

# Open the webcam (index 0 for default webcam)
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    # Capture a frame from the video feed
    (grabbed, frame) = video.read()

    # Check if the frame was successfully grabbed
    if not grabbed:
        print("Error: Unable to grab frame. Exiting.")
        break

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (960, 540))

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(frame, (21, 21), 0)

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Define the lower and upper HSV bounds for fire-like colors
    lower = [18, 50, 50]  # Lower bound for yellow/orange
    upper = [35, 255, 255]  # Upper bound for yellow/orange
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # Create a mask for the fire colors
    mask = cv2.inRange(hsv, lower, upper)

    # Use the mask to extract fire-like regions from the frame
    output = cv2.bitwise_and(frame, frame, mask=mask)

    # Count the number of non-zero pixels in the mask (indicating fire-like colors)
    no_fire_pixels = cv2.countNonZero(mask)

    # If the number of fire pixels exceeds a threshold, report fire
    if int(no_fire_pixels) > 15000:  # Adjust threshold based on the environment
        cv2.putText(frame, "FIRE DETECTED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Print a message whenever fire is detected
        print("Fire detected!")
        
        # Start the alarm if it's not already playing
        if not Alarm_Status:
            Alarm_Status = True
            threading.Thread(target=play_alarm_sound_function, daemon=True).start()

    # Display the output frame and fire detection status
    cv2.imshow("Fire Detection Output", output)
    cv2.imshow("Original Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program.")
        break

# Cleanup: Release the webcam and close OpenCV windows
Alarm_Status = False  # Stop the alarm sound
video.release()
cv2.destroyAllWindows()
