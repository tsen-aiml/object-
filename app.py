import streamlit as st
import cv2
from tempfile import NamedTemporaryFile
from ultralytics import YOLO

def main():
    st.title("Object Detection and Tracking")

    # Upload video file
    video_file = st.file_uploader("Upload video file", type=["mp4"])

    if video_file is not None:
        # Save the uploaded file to a temporary location
        temp_file = NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())

        # Load YOLOv8 model
        model = YOLO('yolov8n.pt')

        # OpenCV VideoCapture object
        cap = cv2.VideoCapture(temp_file.name)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # Define the output video path
        output_video_path = 'output_video.avi'

        # Initialize VideoWriter object
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Read frames
        while True:
            ret, frame = cap.read()

            # Break the loop if no frame is retrieved
            if not ret:
                break

            # Perform object detection
            # Replace this part with the correct method from the ultralytics YOLO API
            # results = model.track(frame, persist=True)

            # Write the frame to the output video
            out.write(frame)

        # Close the temporary file
        temp_file.close()

        # Release resources
        cap.release()
        out.release()

        # Display output video
        st.video(output_video_path)

if __name__ == "__main__":
    main()
