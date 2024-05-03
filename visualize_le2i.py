import cv2

directory = "Data/Le2i/Coffee_room_01"
target_filename = "video (2)"

annotation_file = f"{directory}/Annotation_files/{target_filename}.txt"
video_file = f"{directory}/Videos/{target_filename}.avi"

# Read the annotation file
with open(annotation_file, 'r') as file:
    lines = file.readlines()
    start_frame = int(lines[0])
    end_frame = int(lines[1])
    # fall_frames = [int(line) for line in lines[2:]]

# Read and visualize the video
cap = cv2.VideoCapture(video_file)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("error")
        break
    
    # Check if the current frame is within the fall event
    if frame_count >= start_frame and frame_count <= end_frame:
        # Visualize the fall event on the frame
        # $PLACEHOLDER$
        cv2.putText(frame, f"Frame Count: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Fall Detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Frame Count: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "No Fall", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):  # Increase the delay to slow down the video (50 milliseconds)
        break
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
