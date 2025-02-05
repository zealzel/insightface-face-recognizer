import React, { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";

function App() {
  const webcamRef = useRef(null);
  const [result, setResult] = useState(null);
  const [recognizedFaces, setRecognizedFaces] = useState([]);
  const [name, setName] = useState("");
  const [capturing, setCapturing] = useState(false);
  const [message, setMessage] = useState("");
  const [capturedImages, setCapturedImages] = useState([]);
  const [mode, setMode] = useState("recognition"); // "recognition" 或 "registration"

  const maxCaptures = 12;
  const frameInterval = 833; // 約833毫秒間隔，共捕捉 12 張約 10 秒

  const capture = React.useCallback(async () => {
    let imageSrc;
    if (mode === "registration") {
      // 註冊模式下採用全解析度影像（參照 templates/register.html 的做法）
      imageSrc = webcamRef.current.getScreenshot();
    } else {
      // 辨識模式下使用較低解析度來提高效率
      imageSrc = webcamRef.current.getScreenshot({
        width: 320,
        height: 240,
      });
    }
    // 此處可自行確認影像尺寸
    try {
      const response = await fetch("http://127.0.0.1:5000/recognize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: imageSrc,
          width: 320,
          height: 240,
        }),
      });
      const data = await response.json();
      console.log("data", data);
      setRecognizedFaces(data.recognizedData || []);
    } catch (error) {
      console.error("Error recognizing face:", error);
    }
  }, [webcamRef, mode]);

  const startCapture = () => {
    setCapturing(true);
    setCapturedImages([]);
    let captureCount = 0;
    const captureInterval = setInterval(() => {
      if (captureCount < maxCaptures) {
        // 採用 full resolution 進行截圖
        const imageSrc = webcamRef.current.getScreenshot();
        console.log("imageSrc", imageSrc);
        setCapturedImages((prevImages) => [...prevImages, imageSrc]);
        captureCount++;
        console.log("captureCount", captureCount);
      } else {
        clearInterval(captureInterval);
        setCapturing(false);
        uploadRegistration();
      }
    }, frameInterval);
  };

  const uploadRegistration = async () => {
    try {
      // const response = await fetch("/upload_registration", {
      console.log("capturedImages", capturedImages);
      const response = await fetch(
        "http://127.0.0.1:5000/upload_registration",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: name, images: capturedImages }),
        },
      );
      const data = await response.json();
      setMessage(data.message || "註冊完成");
    } catch (error) {
      console.error("Registration upload failed:", error);
      setMessage("上傳失敗");
    }
  };

  useEffect(() => {
    if (mode === "recognition") {
      const intervalId = setInterval(() => {
        capture();
      }, 200); // 每200毫秒呼叫一次辨識
      return () => clearInterval(intervalId);
    }
  }, [capture, mode]);

  return (
    <div style={{ padding: "20px" }}>
      {/* 模式切換按鈕 */}
      <div style={{ marginBottom: "10px" }}>
        <button onClick={() => setMode("recognition")}>辨識模式</button>
        <button onClick={() => setMode("registration")}>註冊模式</button>
      </div>

      {mode === "recognition" ? (
        // 辨識模式：顯示即時影像及對應方框標記
        <div style={{ position: "relative", display: "inline-block" }}>
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            width={640} // 顯示寬度
            height={480} // 顯示高度
          />
          {recognizedFaces.map((face, index) => {
            // 假設後端傳來的座標根據 320x240 的影像
            const scaleX = 640 / 320;
            const scaleY = 480 / 240;
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
      ) : (
        // 註冊模式：隱藏影像、顯示輸入姓名與按鈕
        <div>
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
          <button onClick={startCapture} disabled={capturing}>
            {capturing ? "錄影中..." : "開始註冊"}
          </button>
          {message && <p>{message}</p>}
          <div style={{ display: "none" }}>
            <Webcam ref={webcamRef} screenshotFormat="image/jpeg" />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
