import React, { useRef, useEffect, useState } from "react";
import VideoStream from "./components/VideoStream";

function App() {
  const videoRef = useRef(null); // 從 VideoStream 存取 video 元素
  const canvasRef = useRef(null); // 用來捕捉影像的隱藏 canvas
  const [recognitionResult, setRecognitionResult] = useState(null);
  const [error, setError] = useState("");

  // 捕捉畫面並呼叫後端辨識 API
  const captureAndRecognize = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      // 若 video 畫面準備好，則設定 canvas 大小
      if (video.videoWidth && video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // 將 canvas 內容轉成 blob (JPEG 格式)，送往後端辨識
        canvas.toBlob((blob) => {
          if (blob) {
            const formData = new FormData();
            formData.append("frame", blob, "frame.jpg");

            fetch("http://127.0.0.1:5000/recognize", {
              method: "POST",
              body: formData,
            })
              .then((response) => response.json())
              .then((data) => {
                setRecognitionResult(data);
                setError("");
              })
              .catch((err) => {
                console.error("辨識錯誤:", err);
                setError(err.message);
              });
          }
        }, "image/jpeg");
      }
    }
  };

  // 每秒呼叫一次 captureAndRecognize 進行辨識
  useEffect(() => {
    const intervalId = setInterval(captureAndRecognize, 1000);
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div>
      <h1>人臉辨識 App</h1>
      <div style={{ position: "relative" }}>
        {/* 將 videoRef 傳給 VideoStream */}
        <VideoStream videoRef={videoRef} />
        {/* 隱藏的 canvas 用來捕捉 video 畫面 */}
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>
      <div>
        <h2>辨識結果</h2>
        {error ? (
          <p style={{ color: "red" }}>錯誤：{error}</p>
        ) : (
          <pre>
            {recognitionResult
              ? JSON.stringify(recognitionResult, null, 2)
              : "無結果"}
          </pre>
        )}
      </div>
    </div>
  );
}

export default App;
