import cv2  
from ultralytics import YOLO

# Load the trained model
model = YOLO(r'D:\human activity\Human Activity.v3i.yolov8\Human_Activity_model.pt')

# Path to the video file
video_path = r'D:\human activity\Human Activity.v3i.yolov8\vid.mp4'  # Change this to your video file path

# Start video capture from the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writers for different formats
output_mp4 = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
output_avi = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
output_mkv = cv2.VideoWriter('output_video.mkv', cv2.VideoWriter_fourcc(*'X264'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to read the frame.")
        break

    # Resize the frame to 640x480
    frame = cv2.resize(frame, (1080, 780))

    # Perform detection
    results = model(frame)

    # Annotate the frame with results
    annotated_frame = results[0].plot()  # This function draws bounding boxes and labels

    # Display the annotated frame
    cv2.imshow('Vehicle Wheel Detection', annotated_frame)

    # Write the annotated frame to all output videos
    output_mp4.write(annotated_frame)
    output_avi.write(annotated_frame)
    output_mkv.write(annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
output_mp4.release()
output_avi.release()
output_mkv.release()
cv2.destroyAllWindows()
