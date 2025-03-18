# Scenarios

## root folder

### on Mac, detect faces in video with cv2 viewer

```bash
python stream.py <video_path>
```

### on Mac, streaming using webcam with cv2 viewer

```bash
python stream_realtime.py
```

## flask-app

### on Mac, streaming using webcam with flask as frontend/backend

```bash
cd flask-app
python app.py
```

### on raspberry pi 4B, streaming using picamera2 with flask as frontend/backend

```bash
cd flask-app
python flaskapp-pi.py
```

### on raspberry pi 4B, streaming using picamera2 with flask as frontend/backend (optimized)

```bash
cd flask-app
python flaskapp-pi-opt.py
```

## insightfacer-react-app

### streaming using webcam with react/flask as frontend/backend

```bash
# frontend
cd insightfacer-react-app/src
npm start
# backend
cd insightfacer-react-app/src/backend
python recognize.py
```

### streaming using webcam with react/flask as frontend/backend (using phone camera)

```bash
ngrok http http://172.20.10.2:3000
```
