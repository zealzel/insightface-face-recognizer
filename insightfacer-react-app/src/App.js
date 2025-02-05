import React, { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";

function App() {
  const webcamRef = useRef(null);
  const [result, setResult] = useState(null);
  const [recognizedFaces, setRecognizedFaces] = useState([]);

  const capture = React.useCallback(async () => {
    // 使用較低解析度進行截圖
    const imageSrc = webcamRef.current.getScreenshot({
      width: 320,
      height: 240,
    });
    const videoWidth = webcamRef.current.video.videoWidth;
    const videoHeight = webcamRef.current.video.videoHeight;

    // console.log("Frontend Image Dimensions:", videoWidth, videoHeight); // 應該是 640x480

    try {
      // const response = await fetch("http://localhost:5000/recognize", {
      const response = await fetch("http://127.0.0.1:5000/recognize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: imageSrc,
          width: 320, // 傳送縮小後的寬度
          height: 240, // 傳送縮小後的高度
        }),
      });
      const data = await response.json();
      console.log("data", data);
      setRecognizedFaces(data.recognizedData || []);
    } catch (error) {
      console.error("Error recognizing face:", error);
    }
  }, [webcamRef]);

  useEffect(() => {
    const intervalId = setInterval(() => {
      capture();
    }, 200); // 調整為每1秒偵測一次
    return () => clearInterval(intervalId);
  }, [capture]);

  return (
    <div style={{ textAlign: "center", position: "relative" }}>
      <h1>Face Recognition</h1>
      <div style={{ position: "relative", display: "inline-block" }}>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width={640} // 顯示的寬度
          height={480} // 顯示的高度
          style={{ position: "relative" }}
        />
        {recognizedFaces.map((face, index) => {
          // 計算縮放比例
          const scaleX = 640 / 320; // 2
          const scaleY = 480 / 240; // 2

          // 根據比例調整坐標
          const top = face.location.top * scaleY;
          const left = face.location.left * scaleX;
          const width = (face.location.right - face.location.left) * scaleX;
          const height = (face.location.bottom - face.location.top) * scaleY;

          return (
            <div
              key={index}
              style={{
                position: "absolute",
                border: "2px solid red",
                top: `${top}px`,
                left: `${left}px`,
                width: `${width}px`,
                height: `${height}px`,
                color: "white",
                backgroundColor: "rgba(0, 0, 0, 0.5)",
                fontSize: "14px",
                fontWeight: "bold",
                textAlign: "center",
                lineHeight: "1",
                padding: "2px",
              }}
            >
              {face.name}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default App;
