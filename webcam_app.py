'''import cv2
import gradio as gr


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame


iface = gr.Interface(
    fn=detect_faces,                 
    inputs=gr.Video(label="Webcamera"),  
    outputs=gr.Video(label="the Output is"),  
    title="Webcam Face Detection", 
    description="Open your webcam to detect faces in real-time."
)

iface.launch()

'''
import cv2
import gradio as gr


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    if frame is None:
        return None
    
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
  
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame

iface = gr.Interface(
    fn=detect_faces,                  
    inputs=gr.Image(),                
    outputs=gr.Image(label="Output"),  
    title="Webcam Face Detection",      
    description="Open your webcam to detect faces."
)


iface.launch()

