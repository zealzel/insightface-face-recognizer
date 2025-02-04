import React, { useState, useRef } from "react";

function Register() {
  const [name, setName] = useState("");
  const [capturing, setCapturing] = useState(false);
  const [message, setMessage] = useState("");
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const capturedImagesRef = useRef([]);

  const maxCaptures = 12;
  const frameInterval = 833; // 約833毫秒間隔，共捕捉 12 張大約 10 秒

  const startRegistration = async () => {
    if (!name) {
      alert("請輸入姓名！");
      return;
    }
    // 重置影像記錄
    capturedImagesRef.current = [];
    setCapturing(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      let captureCount = 0;

      const captureFrame = () => {
        if (captureCount < maxCaptures) {
          const canvas = canvasRef.current;
          const video = videoRef.current;
          if (canvas && video) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL("image/jpeg");
            capturedImagesRef.current.push(dataURL);
          }
          captureCount++;
          setTimeout(captureFrame, frameInterval);
        } else {
          // 停止串流
          if (videoRef.current && videoRef.current.srcObject) {
            const tracks = videoRef.current.srcObject.getTracks();
            tracks.forEach((track) => track.stop());
            videoRef.current.srcObject = null;
          }
          uploadRegistration();
        }
      };
      setTimeout(captureFrame, frameInterval);
    } catch (error) {
      console.error("Error accessing camera:", error);
      setMessage("取得攝影機失敗");
      setCapturing(false);
    }
  };

  const uploadRegistration = async () => {
    try {
      const response = await fetch("/upload_registration", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name, images: capturedImagesRef.current }),
      });
      const data = await response.json();
      setMessage(data.message || "註冊完成");
    } catch (error) {
      console.error("Registration upload failed:", error);
      setMessage("上傳失敗");
    } finally {
      setCapturing(false);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>新增人臉註冊</h1>
      <div>
        姓名：
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          disabled={capturing}
        />
      </div>
      <button onClick={startRegistration} disabled={capturing}>
        {capturing ? "錄影中..." : "開始註冊"}
      </button>
      {message && <p>{message}</p>}

      {/* 隱藏 video 與 canvas 供捕捉用 */}
      <div style={{ display: "none" }}>
        <video ref={videoRef} width="640" height="480" autoPlay muted></video>
        <canvas ref={canvasRef}></canvas>
      </div>
    </div>
  );
}

export default Register;
