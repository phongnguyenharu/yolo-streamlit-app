import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import tempfile
import os
import threading # Thêm thư viện threading
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoTransformerBase # Thư viện webcam deploy

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
# Cần khóa (lock) để tránh xung đột khi nhiều người dùng truy cập
lock = threading.Lock()
detections_container = {"detections": []} # Dùng dict để có thể thay đổi (mutable)

# ---- Class Xử lý Video của Streamlit-WebRTC ----
class YoloVideoTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model
        self.lock = lock
        self.container = detections_container

    def transform(self, frame):
        # Chuyển frame từ WebRTC (PIL Image) sang array (OpenCV)
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
        
        # Trả về khung hình đã vẽ (BGR)
        return annotated_frame

# ---- Cấu hình chính của App ----
st.set_page_config(page_title="YOLOv8 Detection App", layout="wide")
st.title("Ứng dụng Detect Vật thể Real-time với YOLOv8 🚀")
st.write("Phiên bản này sử dụng streamlit-webrtc để có thể deploy.")

# ---- Lựa chọn Model ----
model_path = 'yolov8n.pt'
model = load_yolo_model(model_path)

if model is None:
    st.stop()

# ---- Logic chạy Webcam (Đã thay đổi) ----

st.subheader("Video Feed (Webcam)")
st_data_placeholder = st.empty() # Placeholder cho data, đặt lên trước

# Khởi chạy stream webcam
ctx = webrtc_streamer(
    key="yolo_webcam",
    mode=WebRtcMode.SENDRECV, # Gửi và nhận
    # video_transformer_factory để áp dụng class xử lý của chúng ta
    video_transformer_factory=lambda: YoloVideoTransformer(model), 
    media_stream_constraints={"video": True, "audio": False}, # Chỉ cần video
    async_processing=True, # Xử lý bất đồng bộ
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