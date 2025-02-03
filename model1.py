from flask import Flask, render_template, Response
import cv2
import numpy as np
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

app = Flask(__name__)

# Load the model
model = keras.models.load_model(r"D:\VS_CODE_PROJECTS-NARESH-IT\AI_NIT\sign_detection\Loaded_models\best_model_alphabet_A_to_Z.h5")

background = None
accumulated_weight = 0.5
ROI_top, ROI_bottom, ROI_right, ROI_left = 100, 300, 150, 350
word_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J',
             10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T',
             20:'U', 21:'V', 22:'W', 23:'X', 24:'Y'}

def cal_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment_max_cont)

def generate_frames():
    video_url = "http://192.168.29.197:4747/video"
    cam = cv2.VideoCapture(video_url)
    num_frames = 0
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()

            # Extract the region of interest (ROI) from the frame
            roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
            
            # Update background every few frames (to avoid recalculating on every frame)
            if num_frames < 70 or num_frames % 100 == 0:
                cal_accum_avg(gray_frame, accumulated_weight)
                cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                hand = segment_hand(gray_frame)
                if hand is not None:
                    thresholded, hand_segment = hand
                    
                    # Draw contours around hand
                    cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)
                    
                    # Resize and prepare thresholded image for prediction
                    thresholded = cv2.resize(thresholded, (64, 64))
                    thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                    thresholded = np.reshape(thresholded, (1, thresholded.shape[0], thresholded.shape[1], 3))

                    # Predict the hand sign using the model
                    pred = model.predict(thresholded)
                    predicted_sign = word_dict[np.argmax(pred)]
                    
                    # Display predicted sign on the frame
                    cv2.putText(frame_copy, predicted_sign, (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Draw the ROI rectangle
            cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)

            num_frames += 1
            ret, buffer = cv2.imencode('.jpg', frame_copy)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cam.release()
    cv2.destroyAllWindows()
    
    
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/sign_detection')
def index():
    return render_template('index.html')  # This is for the first sign detection page

@app.route('/sign_detection_2')
def sign_detection_2():
    return render_template('index_2.html')  # This


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
