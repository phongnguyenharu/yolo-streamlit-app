#made bu Haru-Phong

import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import tempfile
import os
import threading
# Import thêm 'av' để chuyển đổi frame
import av 
# Đổi tên class và argument
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoProcessorBase

# Sử dụng cache của Streamlit để tải model chỉ một lần
@st.cache_resource
def load_yolo_model(model_path):
    """
    Tải model YOLOv8 từ đường dẫn.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải model: {e}")
        return None

# ---- Biến toàn cục (để chia sẻ dữ liệu giữa các thread) ----
lock = threading.Lock()
detections_container = {"detections": []} # Dùng dict để có thể thay đổi (mutable)

# ---- Class Xử lý Video (Đã cập nhật) ----
# 1. Đổi tên từ VideoTransformerBase -> VideoProcessorBase
class YoloVideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.lock = lock
        self.container = detections_container

    # 2. Đổi tên hàm từ transform -> recv
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Chuyển frame từ av.VideoFrame (WebRTC) sang array (OpenCV)
        img = frame.to_ndarray(format="bgr24")

        # Chạy detect
        results = self.model(img, stream=True, verbose=False) 

        detections_list = []
        annotated_frame = img.copy() # Phải copy
        
        for r in results:
            annotated_frame = r.plot() 
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                detections_list.append({
                    "Vật thể": class_name,
                    "Độ tự tin": confidence,
                    "Tọa độ (x1, y1, x2, y2)": f"{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}"
                })
        
        # Cập nhật "tín hiệu" vào biến toàn cục một cách an toàn
        with self.lock:
            self.container["detections"] = detections_list
        
        # 3. Trả về khung hình đã vẽ (phải convert về av.VideoFrame)
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# ---- Cấu hình chính của App ----
st.set_page_config(page_title="YOLOv8 Detection App", layout="wide")
st.title("Ứng dụng Detect Vật thể Real-time với YOLOv8 🚀")
st.write("Phiên bản này sử dụng streamlit-webrtc để có thể deploy.")

# ---- Lựa chọn Model ----
model_path = 'yolov8n.pt'
model = load_yolo_model(model_path)

if model is None:
    st.stop()

# ---- Logic chạy Webcam (Đã cập nhật) ----

st.subheader("Video Feed (Webcam)")
st_data_placeholder = st.empty() # Placeholder cho data, đặt lên trước

# Khởi chạy stream webcam
ctx = webrtc_streamer(
    key="yolo_webcam",
    mode=WebRtcMode.SENDRECV,
    # 4. Đổi tên argument: video_transformer_factory -> video_processor_factory
    # 5. Dùng class mới: YoloVideoProcessor
    video_processor_factory=lambda: YoloVideoProcessor(model), 
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.subheader("Tín hiệu (Detections)")

# Vòng lặp để cập nhật bảng "tín hiệu"
# ctx.state.playing cho biết webcam có đang chạy hay không
while ctx.state.playing:
    with lock:
        # Lấy dữ liệu từ biến toàn cục
        detections = detections_container["detections"]
    
    if detections:
        df = pd.DataFrame(detections)
        df["Độ tự tin"] = df["Độ tự tin"].map('{:.2%}'.format) 
        st_data_placeholder.dataframe(df, use_container_width=True)
    else:
        st_data_placeholder.write("Không phát hiện vật thể nào.")
    
    # Refresh 10 lần mỗi giây
    try:
        # Dùng st.rerun() là cách mới và tốt nhất
        st.rerun() 
    except Exception:
        # Fallback cho các phiên bản Streamlit cũ hơn
        st.experimental_rerun()
else:
    st_data_placeholder.empty()
    st.write("Webcam chưa bật. Hãy nhấn 'START' ở khung video trên.")