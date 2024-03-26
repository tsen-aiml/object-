from ultralytics import YOLO
import cv2

# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = './Cars.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = 'output_video_new.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        # detect objects and track them
        results = model.track(frame, persist=True)

        # plot results on the frame
        frame_with_objects = results[0].plot()

        # write the frame with objects to the output video
        out.write(frame_with_objects)

        # visualize the frame with objects
        cv2.imshow('frame', frame_with_objects)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Release VideoCapture and VideoWriter
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
