import cv2

# Path to the input video file
input_video_path = 'uploads/input.mp4'

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get the video frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a VideoWriter object to write the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test.avi', fourcc, 20.0, (frame_width, frame_height))

while True:
    # Read each frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Add text to the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Your Text Here', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Write the frame into the output file
    out.write(frame)

    # (Optional) Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
