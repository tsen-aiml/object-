import streamlit as st
from ultralytics import YOLO
import cv2

def main():
    st.title("Object Detection and Tracking with YOLOv8")

    # Upload video file
    video_file = st.file_uploader("Upload video file", type=["mp4"])

    if video_file is not None:
        # Load YOLOv8 model
        model = YOLO('yolov8n.pt')

        # OpenCV VideoCapture object
        cap = cv2.VideoCapture('./Zebra and Giraffe.mp4')

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))

        # Read frames
        while True:
            ret, frame = cap.read()

            # Break the loop if no frame is retrieved
            if not ret:
                break

            # Detect objects and track them
            results = model.track(frame, persist=True)

            # Plot results on the frame
            frame_with_objects = results[0].plot()

            # Write the frame with objects to the output video
            out.write(frame_with_objects)

            # Display frame
            st.image(frame_with_objects, channels="BGR")

        # Release resources
        cap.release()
        out.release()

if __name__ == "__main__":
    main()
