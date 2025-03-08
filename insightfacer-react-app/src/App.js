import React, { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";

const FACING_MODE_USER = "user";
const FACING_MODE_ENVIRONMENT = "environment";
const WIDTH = 640;
const HEIGHT = 480;
// const WIDTH = 1280;
// const HEIGHT = 760;
const RECOG_WIDTH = 320;
const RECOG_HEIGHT = 240;

function App() {
  const webcamRef = useRef(null);
  const [result, setResult] = useState(null);
  const [recognizedFaces, setRecognizedFaces] = useState([]);
  const [name, setName] = useState("");
  const [capturing, setCapturing] = useState(false);
  const [message, setMessage] = useState("");
  const [capturedImages, setCapturedImages] = useState([]);
  const [mode, setMode] = useState("recognition"); // "recognition" 或 "registration"
  const capturedImagesRef = useRef([]);
  const [facingMode, setFacingMode] = React.useState(FACING_MODE_USER);

  const maxCaptures = 12;
  const frameInterval = 833; // 約833毫秒間隔，共捕捉 12 張約 10 秒

  const capture = React.useCallback(async () => {
    let imageSrc;
    imageSrc = webcamRef.current.getScreenshot();
    if (mode === "registration") {
      // 註冊模式：採用全解析度截圖，不傳入 width/height 參數
      imageSrc = webcamRef.current.getScreenshot();
    } else {
      // 辨識模式：採用縮小解析度以加快辨識速度
      imageSrc = webcamRef.current.getScreenshot({
        width: RECOG_WIDTH,
        height: RECOG_HEIGHT,
      });
    }
    // 此處可自行確認影像尺寸
    try {
      // const response = await fetch("http://127.0.0.1:5000/recognize", {
      const response = await fetch("/recognize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: imageSrc,
          width: RECOG_WIDTH,
          height: RECOG_HEIGHT,
        }),
      });
      const data = await response.json();
      console.log("data", data);
      setRecognizedFaces(data.recognizedData || []);
    } catch (error) {
      console.error("Error recognizing face:", error);
    }
  }, [webcamRef, mode]);

  let videoConstraints = {
    facingMode: facingMode,
    width: WIDTH,
    height: HEIGHT,
  };

  const handleClick = React.useCallback(() => {
    console.log("handleClick");
    setFacingMode((prevState) =>
      prevState === FACING_MODE_USER
        ? FACING_MODE_ENVIRONMENT
        : FACING_MODE_USER,
    );
  }, []);

  const startCapture = () => {
    setCapturing(true);
    setCapturedImages([]);
    capturedImagesRef.current = [];
    let captureCount = 0;
    const captureInterval = setInterval(() => {
      if (captureCount < maxCaptures) {
        // 註冊模式下使用 full resolution 截圖（不傳入尺寸參數）
        const imageSrc = webcamRef.current.getScreenshot();
        console.log("imageSrc", imageSrc);
        capturedImagesRef.current.push(imageSrc);
        setCapturedImages([...capturedImagesRef.current]);
        captureCount++;
        console.log("captureCount", captureCount);
      } else {
        clearInterval(captureInterval);
        setCapturing(false);
        // 上傳時直接使用 ref 內的圖片陣列
        uploadRegistration(capturedImagesRef.current);
      }
    }, frameInterval);
  };

  const uploadRegistration = async (imagesArray) => {
    try {
      console.log("capturedImages", imagesArray);
      const response = await fetch(
        "http://127.0.0.1:5000/upload_registration",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: name, images: imagesArray }),
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
      }, 100); // 每100毫秒呼叫一次辨識
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
            videoConstraints={videoConstraints}
          />
          <button onClick={handleClick}>Switch camera</button>
          {recognizedFaces.map((face, index) => {
            // Get the video element from react-webcam
            const video = webcamRef.current && webcamRef.current.video;
            // Use the actual rendered size if available; fallback to your constants otherwise
            const previewWidth = video ? video.clientWidth : WIDTH;
            const previewHeight = video ? video.clientHeight : HEIGHT;
            // Calculate dynamic scale factors based on the recognition resolution (320x240)
            const scaleX = previewWidth / RECOG_WIDTH;
            const scaleY = previewHeight / RECOG_HEIGHT;

            const top = face.location.top * scaleY;
            const left = face.location.left * scaleX;
            const boxWidth =
              (face.location.right - face.location.left) * scaleX;
            const boxHeight =
              (face.location.bottom - face.location.top) * scaleY;

            return (
              <div
                key={index}
                style={{
                  position: "absolute",
                  border: "2px solid red",
                  top: `${top}px`,
                  left: `${left}px`,
                  width: `${boxWidth}px`,
                  height: `${boxHeight}px`,
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
        // 註冊模式：隱藏影像、顯示輸入姓名與按鈕，以及 Debug 區塊（顯示暫存的拍攝圖片）
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
          {/* Debug 區塊：顯示捕捉到的圖片 */}
          {capturedImages.length > 0 && (
            <div style={{ marginTop: "20px" }}>
              <h2>Debug: Captured Images</h2>
              <div style={{ display: "flex", flexWrap: "wrap" }}>
                {capturedImages.map((img, idx) => (
                  <img
                    key={idx}
                    src={img}
                    alt={`capture ${idx}`}
                    style={{
                      width: "160px",
                      height: "120px",
                      margin: "5px",
                      border: "1px solid #ccc",
                    }}
                  />
                ))}
              </div>
            </div>
          )}
          <div
            style={{
              width: "1280px",
              height: "720px",
              opacity: 0,
            }}
          >
            <Webcam
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              videoConstraints={{
                width: { ideal: 1280 },
                height: { ideal: 720 },
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
