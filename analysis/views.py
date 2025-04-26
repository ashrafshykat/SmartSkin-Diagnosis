from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
import cv2

from predictor import predict_from_upload

from predictor import predict_from_url


def index(request):
    return render(request, 'analysis/index.html', {})

UPLOAD_DIR = "D:/WebDjango/"

def analyze(request):
    if request.method == 'POST':
        image_url = request.POST.get('image_url')
        if image_url:
            prediction = predict_from_url(image_url)
            return render(request, 'analysis/analyze_result.html', {'prediction': prediction})
    return HttpResponse("<h2>Error: Invalid request</h2>", status=400)

def upload(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']

        prediction = predict_from_upload(uploaded_file)
        return render(request, 'analysis/upload_result.html', {'prediction': prediction})
    return HttpResponse("<h2>Error: Invalid request</h2>", status=400)

def generate_frames():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def realtime(request):
    return StreamingHttpResponse(generate_frames(), content_type="multipart/x-mixed-replace; boundary=frame")