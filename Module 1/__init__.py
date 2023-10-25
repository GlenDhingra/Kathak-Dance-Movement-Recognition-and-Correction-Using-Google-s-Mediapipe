from flask import Flask, redirect, url_for, request, render_template, Response
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

def generate_frames(gestureName):
    while True:
        with open("./models/"+gestureName+'.pkl','rb') as f:
            model_body = pickle.load(f)
        ret,frame = cap.read()
        with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        

            image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image.flags.writeable=False
            
            results = holistic.process(image)

            image.flags.writeable=True
            image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            
            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )
            
            try:
                #cv2.rectangle(image,(0,30),(len(image[0]), 60),(245,117,16),-1)
                rHand = results.right_hand_landmarks.landmark
                rHand_row = list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility]for landmark in rHand]).flatten())   
                row = rHand_row

                
                # Hands
                X4 = pd.DataFrame([row])
                body_hands_class = model_body.predict(X4)[0]
                body_hands_prob = model_body.predict_proba(X4)[0]

                cv2.putText(image, str(gestureName), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(body_hands_prob[0]) , (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
            except Exception as e:
                # print(e)
                pass

            ret, buffer = cv2.imencode('.jpg', image)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
            b'Content-Type: frame/jpeg\r\n\r\n' + img + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html", title="Home")
 
@app.route('/hands/')
def hand():
    # return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return render_template("hand_home.html", title="Hand Gesture Home")

@app.route('/handGesture/', methods=['POST'])
def handGesture():
    gestureName = ""
    for i in request.form:
        gestureName = i
    print("here")
    return render_template("hand.html", title="Hand Gesture",data=gestureName)

@app.route('/video_feed')
def video_feed():
    gestureName = request.args.get("gestureName")
    return Response(generate_frames(gestureName), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
 
    app.run()