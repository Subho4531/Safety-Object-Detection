import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
import os
import io

# Fixed model configuration
MODEL_PATH = "models/best.pt"
DEFAULT_CONFIDENCE = 0.25

# Set page configuration
st.set_page_config(
    page_title="Safety Object Detection",
    page_icon="⛑️",
    layout="wide"
)

# Title and description
st.title("Safety Object Detection")
st.markdown("Upload a video or image to detect safety-related objects using the built-in YOLO11 model")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Safety Settings")
    
    st.markdown("Using fixed model and default confidence settings.")
    
    # Display options
    show_fps = st.checkbox("Show FPS", value=True)
    show_labels = st.checkbox("Show Labels", value=True)
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    
    # Model info section
    st.divider()
    st.markdown("### Model Information")
    
    if os.path.exists(MODEL_PATH):
        st.success("✅ Model found!")
        
        # Optional: Display model metrics if available
        metrics_path = MODEL_PATH.replace("weights/best.pt", "results.csv")
        if os.path.exists(metrics_path):
            try:
                import pandas as pd
                metrics_df = pd.read_csv(metrics_path)
                if not metrics_df.empty:
                    st.markdown("**Training Metrics:**")
                    last_row = metrics_df.iloc[-1]
                    if 'metrics/mAP50(B)' in metrics_df.columns:
                        st.markdown(f"**mAP@50:** {last_row['metrics/mAP50(B)']:.3f}")
                    if 'metrics/mAP50-95(B)' in metrics_df.columns:
                        st.markdown(f"**mAP@50-95:** {last_row['metrics/mAP50-95(B)']:.3f}")
            except:
                pass
    else:
        st.warning(f"⚠️ Fixed model not found at {MODEL_PATH}. Please place the weights there.")
    
    st.divider()
    st.markdown("### Detection Statistics")
    stats_placeholder = st.empty()

# st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"] {display: none;}
#     [data-testid="collapsedControl"] {display: none;}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# Initialize session state
if 'detections_history' not in st.session_state:
    st.session_state.detections_history = []
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'class_names' not in st.session_state:
    st.session_state.class_names = {}

# Load custom YOLO model
@st.cache_resource
def load_custom_model():
    try:
        _model_path = MODEL_PATH
        if not os.path.exists(_model_path):
            # Try to find the model in common locations
            possible_paths = [
                _model_path,
                f"kaggle/working/{_model_path}",
                f"/kaggle/working/{_model_path}",
                _model_path.replace("space_mission/", ""),
                "best.pt",
                "space_mission/yolo11m_run/weights/best.pt"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    _model_path = path
                    st.sidebar.info(f"Found model at: {path}")
                    break
        
        model = YOLO(_model_path)
        
        # Try to get class names from the model
        try:
            if hasattr(model, 'names') and model.names:
                st.session_state.class_names = model.names
                st.sidebar.success(f"Model loaded with {len(model.names)} classes")
            else:
                # Try to load from data yaml
                import yaml
                yaml_path = _model_path.replace('weights/best.pt', 'args.yaml')
                if os.path.exists(yaml_path):
                    with open(yaml_path, 'r') as f:
                        args = yaml.safe_load(f)
                        if 'data' in args:
                            data_yaml = args['data']
                            if os.path.exists(data_yaml):
                                with open(data_yaml, 'r') as df:
                                    data = yaml.safe_load(df)
                                    if 'names' in data:
                                        st.session_state.class_names = data['names']
                                        st.sidebar.success(f"✅ Model loaded with {len(data['names'])} classes")
        except:
            st.session_state.class_names = {}
            st.sidebar.warning("⚠️ Could not load class names")
        
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

# Load the model
model = load_custom_model()

# Define color palette for different classes
def get_class_color(class_id, total_classes):
    # Generate distinct colors for each class
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
    ]
    return colors[class_id % len(colors)]

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Media Preview")
    input_mode = st.radio("Input Type", ["Video", "Image"], horizontal=True)
    
    uploaded_video = None
    uploaded_image = None
    
    if input_mode == "Video":
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            key="video_uploader"
        )
    else:
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            key="image_uploader"
        )
    
    media_placeholder = st.empty()
    
    # Control buttons
    col1_1, col1_2, col1_3 = st.columns(3)
    start_btn = False
    stop_btn = False
    detect_image_btn = False
    with col1_1:
        if input_mode == "Video":
            start_btn = st.button("▶️ Start Detection", type="primary", 
                                 disabled=not uploaded_video or model is None)
        else:
            detect_image_btn = st.button("🔍 Run Detection", type="primary",
                                         disabled=not uploaded_image or model is None)
    with col1_2:
        if input_mode == "Video":
            stop_btn = st.button("⏹️ Stop", disabled=not st.session_state.processing)
        else:
            st.button("⏸️ Stop", disabled=True)
    with col1_3:
        clear_btn = st.button("🗑️ Clear Results")

with col2:
    st.subheader("Detected Objects")
    
    # Real-time detection list
    detection_list = st.empty()
    
    # Object statistics
    st.subheader("Object Statistics")
    stats_chart = st.empty()

# Function to process video
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    last_time = time.time()
    
    while st.session_state.processing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference with custom model
        results = model(frame_rgb, conf=DEFAULT_CONFIDENCE, verbose=False)[0]
        
        # Extract detections
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, box)
                
                # Get class name
                if st.session_state.class_names:
                    class_name = st.session_state.class_names.get(cls_id, f"Class_{cls_id}")
                else:
                    class_name = f"Class_{cls_id}"
                
                # Store detection
                detection = {
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': (x1, y1, x2, y2),
                    'frame': frame_count,
                    'class_id': int(cls_id)
                }
                detections.append(detection)
                
                # Get color for this class
                total_classes = len(st.session_state.class_names) if st.session_state.class_names else 10
                color = get_class_color(cls_id, total_classes)
                
                # Draw bounding box
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                
                # Create label
                label = f"{class_name}"
                if show_confidence:
                    label += f" {conf:.2f}"
                
                # Draw label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Ensure label stays within frame
                label_y = max(y1 - 10, label_height + 5)
                
                # Draw background rectangle for label
                cv2.rectangle(
                    frame_rgb,
                    (x1, label_y - label_height - 5),
                    (x1 + label_width + 10, label_y + 5),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame_rgb,
                    label,
                    (x1 + 5, label_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )
        
        # Update session state
        st.session_state.current_frame = frame_rgb
        st.session_state.detections_history.extend(detections)
        
        # Calculate FPS
        current_time = time.time()
        fps_display = 1 / (current_time - last_time) if frame_count > 0 else 0
        last_time = current_time
        
        # Display FPS on frame
        if show_fps:
            cv2.putText(
                frame_rgb,
                f"FPS: {fps_display:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        
        # Display frame count
        cv2.putText(
            frame_rgb,
            f"Frame: {frame_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
                (0, 255, 0),
                2
            )
        
        # Convert to PIL Image for Streamlit
        pil_image = Image.fromarray(frame_rgb)
        
        # Update video display
        media_placeholder.image(pil_image, caption="Live Detection", use_container_width=True)
        
        render_detection_list(detections)
        
        # Update statistics
        update_statistics()
        
        frame_count += 1
    
    cap.release()
    st.session_state.processing = False

# Function to render detection list for current frame/image
def render_detection_list(detections):
    if detections:
        unique_objects = {}
        for det in detections:
            class_name = det['class']
            if class_name not in unique_objects:
                unique_objects[class_name] = {
                    'count': 1,
                    'max_conf': det['confidence'],
                    'class_id': det['class_id']
                }
            else:
                unique_objects[class_name]['count'] += 1
                if det['confidence'] > unique_objects[class_name]['max_conf']:
                    unique_objects[class_name]['max_conf'] = det['confidence']
        
        detection_text = "### Current Frame Detections:\n"
        for obj, data in sorted(unique_objects.items()):
            detection_text += f"- **{obj}**: {data['count']} "
            if show_confidence:
                detection_text += f"(max: {data['max_conf']:.2f})"
            detection_text += "\n"
        
        detection_list.markdown(detection_text)
    else:
        detection_list.empty()


# Function to update statistics
def update_statistics():
    if st.session_state.detections_history:
        # Count objects by class
        object_counts = {}
        for det in st.session_state.detections_history:
            cls_name = det['class']
            object_counts[cls_name] = object_counts.get(cls_name, 0) + 1
        
        # Update sidebar statistics
        total_frames = len(set(d['frame'] for d in st.session_state.detections_history))
        total_objects = len(st.session_state.detections_history)
        
        stats_text = f"**Total Frames:** {total_frames}\n"
        stats_text += f"**Total Detections:** {total_objects}\n\n"
        stats_text += "**Top Objects Detected:**\n"
        
        for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            stats_text += f"• {obj}: {count}\n"
        
        stats_placeholder.markdown(stats_text)
        
        # Create bar chart
        if len(object_counts) > 0:
            import pandas as pd
            chart_data = pd.DataFrame({
                "Object": list(object_counts.keys()),
                "Count": list(object_counts.values())
            })
            # Sort by count for better visualization
            chart_data = chart_data.sort_values("Count", ascending=False).head(10)
            stats_chart.bar_chart(chart_data, x="Object", y="Count")

# Function to process an image
def process_image(image_bytes, model):
    st.session_state.processing = False
    st.session_state.detections_history = []
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    frame_rgb = np.array(image)
    results = model(frame_rgb, conf=DEFAULT_CONFIDENCE, verbose=False)[0]
    detections = []
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = st.session_state.class_names.get(cls_id, f"Class_{cls_id}") if st.session_state.class_names else f"Class_{cls_id}"
            detection = {
                'class': class_name,
                'confidence': float(conf),
                'bbox': (x1, y1, x2, y2),
                'frame': 0,
                'class_id': int(cls_id)
            }
            detections.append(detection)
            color = get_class_color(cls_id, len(st.session_state.class_names) if st.session_state.class_names else 10)
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}"
            if show_confidence:
                label += f" {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(y1 - 10, label_height + 5)
            cv2.rectangle(frame_rgb, (x1, label_y - label_height - 5), (x1 + label_width + 10, label_y + 5), color, -1)
            cv2.putText(frame_rgb, label, (x1 + 5, label_y), font, font_scale, (255, 255, 255), thickness)
    
    st.session_state.current_frame = frame_rgb
    st.session_state.detections_history.extend(detections)
    pil_image = Image.fromarray(frame_rgb)
    media_placeholder.image(pil_image, caption="Detection Result", use_container_width=True)
    render_detection_list(detections)
    update_statistics()


# Button handlers
if input_mode == "Video" and start_btn and uploaded_video and model:
    st.session_state.processing = True
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_video.getvalue())
        video_path = tmp_file.name
    
    # Process video in a separate thread (simplified for demo)
    process_video(video_path, model)

if input_mode == "Image" and detect_image_btn and uploaded_image and model:
    process_image(uploaded_image.getvalue(), model)

if stop_btn:
    st.session_state.processing = False
    st.rerun()

if clear_btn:
    st.session_state.detections_history = []
    st.session_state.current_frame = None
    media_placeholder.empty()
    detection_list.empty()
    stats_placeholder.empty()
    stats_chart.empty()
    st.rerun()

# Footer with space mission specific info
st.divider()
st.markdown("""
---
**About This Model:**
- **Model**: Fixed YOLO11m weights finetuned model on virtual space mission dataset
- **Training**: 50 epochs with batch size 16
- **Classes**: Detects various Safety-related objects (Oxygen tanks, Nitrogen tanks, First aid kits, Fire extinguishers, etc.)
- **Performance**: mAP@50 accuracy around 95% on validation set

**How to use:**
1. Upload a video or image containing space objects
2. Click "Start Detection" for videos or "Run Detection" for images

**Expected Classes:**
Based on your training configuration, this model detects .
""")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)