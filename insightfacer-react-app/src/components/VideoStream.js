import React, { useRef, useEffect } from "react";

// 如果 props 中傳入 videoRef，就使用傳入的參考，否則採用內部 ownVideoRef
const VideoStream = ({ videoRef: externalVideoRef }) => {
  const ownVideoRef = useRef(null);
  const videoRef = externalVideoRef || ownVideoRef;

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => {
        console.error("無法取得影片串流:", err);
      });
  }, [videoRef]);

  return (
    <div>
      <h2>影片串流</h2>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        controls
        style={{ width: "100%" }}
      />
    </div>
  );
};

export default VideoStream;
