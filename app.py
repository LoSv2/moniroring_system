"""
Streamlit приложение для мониторинга дисциплины на занятиях
Детектирует нарушения: сон, телефон, еда/напитки
"""
import os
import cv2
import time
import pickle
import tempfile
import numpy as np
import streamlit as st
from datetime import datetime
from pathlib import Path

# Импорт модулей
from modules.detection import ViolationDetector
from modules.face_recognition import FaceRecognizer
from modules.video_processor import VideoProcessor
from modules.detection_logic import (
    process_frame_for_detection_correct,
    draw_detections_with_boxes,
    draw_sleep_indicator,
    load_face_resources,
    analyze_video_segment
)

# ═══════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ STREAMLIT
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Монитор Дисциплины",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомный стиль
st.markdown("""
<style>
    /* Основные переменные */
    :root {
        --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --card-bg: rgba(255, 255, 255, 0.95);
        --text-primary: #2d3748;
        --text-secondary: #4a5568;
        --accent: #667eea;
        --success: #48bb78;
        --warning: #ed8936;
        --danger: #f56565;
        --info: #4299e1;
    }

    /* Заголовок */
    .main-title {
        background: var(--bg-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 30px;
        letter-spacing: -0.5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: fadeInDown 0.8s ease-out;
    }

    /* Карточки для метрик */
    .metric-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: 700;
        color: var(--accent);
        line-height: 1.2;
    }
    .metric-label {
        font-size: 1em;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Бейджи нарушений */
    .violation-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 50px;
        color: white;
        font-weight: 600;
        font-size: 0.9em;
        margin: 4px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .violation-badge:hover {
        transform: scale(1.05);
    }
    .sleeping { background: linear-gradient(135deg, #f56565 0%, #c53030 100%); }
    .phone { background: linear-gradient(135deg, #ed8936 0%, #c05621 100%); }
    .food { background: linear-gradient(135deg, #48bb78 0%, #2f855a 100%); }
    .bottle { background: linear-gradient(135deg, #4299e1 0%, #2b6cb0 100%); }

    /* Стилизация боковой панели */
    .css-1d391kg, .css-1lcbmhc {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
    .sidebar-header {
        font-size: 1.5em;
        font-weight: 600;
        color: var(--accent);
        margin-bottom: 20px;
        text-align: center;
    }

    /* Кнопки */
    .stButton > button {
        background: var(--bg-gradient);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 10px 25px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Контейнер для видео */
    .video-container {
        max-height: 500px;
        overflow: auto;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }

    /* Анимации */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Адаптация для мобильных */
    @media (max-width: 768px) {
        .main-title { font-size: 2em; }
        .metric-card { padding: 15px; }
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# ИНИЦИАЛИЗАЦИЯ SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

if 'detector' not in st.session_state:
    st.session_state.detector = None

if 'face_recognizer' not in st.session_state:
    st.session_state.face_recognizer = None

if 'video_processor' not in st.session_state:
    st.session_state.video_processor = VideoProcessor()

if 'violations_log' not in st.session_state:
    st.session_state.violations_log = []

if 'processing' not in st.session_state:
    st.session_state.processing = False

# ═══════════════════════════════════════════════════════════════════════
# ФУНКЦИИ КЭШИРОВАНИЯ
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_detector(model_path):
    """Кэширует детектор нарушений"""
    return ViolationDetector(model_path)

@st.cache_resource
def load_face_recognizer(db_path):
    """Кэширует распознаватель лиц"""
    return FaceRecognizer(db_path if os.path.exists(db_path) else None)

# ═══════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОБРАБОТКИ
# ═══════════════════════════════════════════════════════════════════════

def process_frame_for_detection(current_time, detections_in_frame, sleep_start_time, sleep_buffer):
    """
    ПРАВИЛЬНАЯ обработка кадра для получения confirmed_violations.
    Использует логику из standalone скрипта.
    
    Args:
        current_time: текущее время
        detections_in_frame: set названий обнаруженных классов {'sleeping', 'phone', ...}
        sleep_start_time: время начала обнаружения сна
        sleep_buffer: буфер подтверждения сна в секундах
    
    Returns:
        (confirmed_violations, new_sleep_start_time, new_last_detection_time)
    """
    return process_frame_for_detection_correct(current_time, detections_in_frame, sleep_start_time, sleep_buffer)


# ═══════════════════════════════════════════════════════════════════════
# ФУНКЦИИ ОБРАБОТКИ ВИДЕО (определены перед использованием)
# ═══════════════════════════════════════════════════════════════════════

def process_webcam(frame_skip=2, buffer_seconds=10, sleep_buffer=10, face_db_path="students.pkl", face_similarity=0.5):
    """Обработка потока веб-камеры"""
    try:
        st.info("Инициализация веб-камеры... (может занять 3-5 секунд)")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            st.error("Не удалось подключиться к веб-камере. Проверьте её подключение.")
            st.session_state.processing = False
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        st.session_state.video_processor.setup_output_dirs()
        
        metrics = {
            'total_frames': 0,
            'violations': {},
            'recording': False,
            'frames_processed': 0
        }
        
        frame_count = 0
        sleep_start_time = None
        last_detection_time = None
        last_recording_end_time = None
        recording = False
        rec_violations = set()
        current_segment_path = None
        last_detections = {}
        last_confirmed_violations = set()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        frame_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
        metrics_placeholder = st.empty()
        
        st.success("Веб-камера готова!")
        stop_button = st.button("⏹ Остановить трансляцию", key="webcam_stop", use_container_width=True)
        
        max_frames = 9000
        
        while cap.isOpened() and frame_count < max_frames and not stop_button and st.session_state.processing:
            ret, frame = cap.read()
            if not ret:
                st.warning("Ошибка при чтении с веб-камеры")
                break
            
            frame_count += 1
            metrics['total_frames'] = frame_count
            
            annotated_frame = frame.copy()
            current_time = time.time()
            confirmed_violations = set()
            
            if frame_count % frame_skip == 0:
                detections, _ = st.session_state.detector.detect_frame(frame, draw_boxes=False)
                
                for class_name in detections:
                    if class_name not in metrics['violations']:
                        metrics['violations'][class_name] = 0
                    metrics['violations'][class_name] += 1
                
                confirmed_violations, sleep_start_time, detection_time = process_frame_for_detection(
                    current_time, set(detections.keys()), sleep_start_time, sleep_buffer
                )
                
                if confirmed_violations:
                    if not recording:
                        segments_dir, _ = st.session_state.video_processor.setup_output_dirs()
                        filename = st.session_state.video_processor.generate_segment_filename()
                        current_segment_path = os.path.join(segments_dir, filename)
                        
                        st.session_state.video_processor.start_recording(
                            current_segment_path,
                            (width, height),
                            fps
                        )
                        recording = True
                        rec_violations = set(confirmed_violations)
                    else:
                        rec_violations.update(confirmed_violations)
            
            if last_detections and last_confirmed_violations:
                violations_to_draw = last_confirmed_violations & set(last_detections.keys())
                for class_name in violations_to_draw:
                    if class_name in last_detections:
                        boxes_list = last_detections[class_name]
                        for box_info in boxes_list:
                            x1, y1, x2, y2 = map(int, box_info['box'])
                            conf = box_info['conf']
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{class_name} {conf:.2f}"
                            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if 'sleeping' in last_detections:
                if sleep_start_time is not None:
                    time_elapsed = current_time - sleep_start_time
                    if time_elapsed >= sleep_buffer:
                        cv2.putText(annotated_frame, "SLEEP CONFIRMED", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        time_left = sleep_buffer - time_elapsed
                        cv2.putText(annotated_frame, f"Sleep Buffer: {time_left:.1f}s", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
            if recording:
                cv2.circle(annotated_frame, (30, 30), 10, (0, 0, 255), -1)
                st.session_state.video_processor.write_frame(annotated_frame)
            
            if recording and last_detection_time:
                if (current_time - last_detection_time) > buffer_seconds:
                    st.session_state.video_processor.stop_recording()
                    recording = False
                    
                    st.session_state.violations_log.append({
                        'path': current_segment_path,
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'violation': ", ".join(sorted(rec_violations)),
                        'student': 'Обработка...',
                        'confidence': 'N/A'
                    })
                    last_confirmed_violations = set()
            
            frame_placeholder.image(annotated_frame, channels="BGR")
            metrics_placeholder.write(f"Обработано кадров: {metrics['total_frames']} | Нарушений: {len(st.session_state.violations_log)}")
            progress = min(frame_count / max_frames, 1.0)
            progress_bar.progress(progress)
        
        cap.release()
        if recording:
            st.session_state.video_processor.stop_recording()
        
        if st.session_state.violations_log:
            st.info("Анализ лиц в обнаруженных нарушениях...")
            
            if st.session_state.face_recognizer is None and os.path.exists(face_db_path):
                st.session_state.face_recognizer = load_face_recognizer(face_db_path)
            
            if st.session_state.face_recognizer and st.session_state.face_recognizer.is_database_available():
                progress_face = st.progress(0)
                face_status = st.empty()
                
                violations_to_process = [
                    i for i, v in enumerate(st.session_state.violations_log) 
                    if v['student'] == 'Обработка...'
                ]
                
                for idx, i in enumerate(violations_to_process):
                    face_status.text(f"Обработка нарушения {idx + 1}/{len(violations_to_process)}...")
                    violation_path = st.session_state.violations_log[i]['path']
                    
                    try:
                        name, score, face_path = st.session_state.face_recognizer.analyze_video_segment(
                            violation_path,
                            face_similarity=face_similarity
                        )
                        st.session_state.violations_log[i]['student'] = name
                        st.session_state.violations_log[i]['confidence'] = f"{score:.0%}"
                    except Exception as e:
                        st.error(f"Ошибка при анализе {Path(violation_path).name}: {str(e)}")
                        st.session_state.violations_log[i]['student'] = "Не опознан"
                        st.session_state.violations_log[i]['confidence'] = "Ошибка анализа"
                    
                    progress_face.progress((idx + 1) / len(violations_to_process))
                
                face_status.empty()
                progress_face.empty()
            else:
                for i, violation in enumerate(st.session_state.violations_log):
                    if violation['student'] == 'Обработка...':
                        st.session_state.violations_log[i]['student'] = "Не опознан"
                        st.session_state.violations_log[i]['confidence'] = "Нет БД"
        
        st.success(f"Обработка завершена! Обнаружено {len(st.session_state.violations_log)} нарушений.")
        st.session_state.processing = False
    
    except Exception as e:
        st.error(f"Ошибка при обработке веб-камеры: {e}")
        st.session_state.processing = False

def process_video_file(video_path, frame_skip=2, buffer_seconds=10, sleep_buffer=10, face_db_path="students.pkl", face_similarity=0.5):
    """Обработка видеофайла"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        st.session_state.video_processor.setup_output_dirs()
        
        metrics = {
            'total_frames': 0,
            'violations': {},
            'recording': False,
            'frames_processed': 0
        }
        
        frame_count = 0
        sleep_start_time = None
        last_detection_time = None
        last_recording_end_time = None
        recording = False
        rec_violations = set()
        current_segment_path = None
        
        last_detections = {}
        last_confirmed_violations = set()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        frame_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
        metrics_placeholder = st.empty()
        
        while cap.isOpened() and st.session_state.processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            metrics['total_frames'] = frame_count
            annotated_frame = frame.copy()
            current_time = time.time()
            confirmed_violations = set()
            
            if frame_count % frame_skip == 0:
                detections, _ = st.session_state.detector.detect_frame(frame, draw_boxes=False)
                
                for class_name in detections:
                    if class_name not in metrics['violations']:
                        metrics['violations'][class_name] = 0
                    metrics['violations'][class_name] += 1
                
                confirmed_violations, sleep_start_time, detection_time = process_frame_for_detection(
                    current_time, set(detections.keys()), sleep_start_time, sleep_buffer
                )
                
                last_detections = detections
                last_confirmed_violations = confirmed_violations
                
                if confirmed_violations and detection_time:
                    last_detection_time = detection_time
            
            if confirmed_violations:
                if not recording:
                    segments_dir, _ = st.session_state.video_processor.setup_output_dirs()
                    filename = st.session_state.video_processor.generate_segment_filename()
                    current_segment_path = os.path.join(segments_dir, filename)
                    
                    st.session_state.video_processor.start_recording(
                        current_segment_path,
                        (width, height),
                        fps
                    )
                    recording = True
                    rec_violations = set(confirmed_violations)
                else:
                    rec_violations.update(confirmed_violations)
            
            if last_detections:
                for class_name, boxes_list in last_detections.items():
                    for box_info in boxes_list:
                        x1, y1, x2, y2 = map(int, box_info['box'])
                        conf = box_info['conf']
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if 'sleeping' in last_detections:
                if sleep_start_time is not None:
                    time_elapsed = current_time - sleep_start_time
                    if time_elapsed >= sleep_buffer:
                        cv2.putText(annotated_frame, "SLEEP CONFIRMED", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        time_left = sleep_buffer - time_elapsed
                        cv2.putText(annotated_frame, f"Sleep Buffer: {time_left:.1f}s", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
            if recording:
                cv2.circle(annotated_frame, (30, 30), 10, (0, 0, 255), -1)
                st.session_state.video_processor.write_frame(annotated_frame)
            
            if recording and last_detection_time:
                if (current_time - last_detection_time) > buffer_seconds:
                    st.session_state.video_processor.stop_recording()
                    recording = False
                    
                    st.session_state.violations_log.append({
                        'path': current_segment_path,
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'violation': ", ".join(sorted(rec_violations)),
                        'student': 'Обработка...',
                        'confidence': 'N/A'
                    })
                    last_confirmed_violations = set()
            
            frame_placeholder.image(annotated_frame, channels="BGR")
            metrics_placeholder.write(f"Обработано: {metrics['total_frames']} кадров | Нарушений: {len(st.session_state.violations_log)}")
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_frames > 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
        
        cap.release()
        if recording:
            st.session_state.video_processor.stop_recording()
        
        if st.session_state.violations_log:
            st.info("Анализ лиц в обнаруженных нарушениях...")
            
            if st.session_state.face_recognizer is None and os.path.exists(face_db_path):
                st.session_state.face_recognizer = load_face_recognizer(face_db_path)
            
            if st.session_state.face_recognizer and st.session_state.face_recognizer.is_database_available():
                progress_face = st.progress(0)
                face_status = st.empty()
                
                violations_to_process = [
                    i for i, v in enumerate(st.session_state.violations_log) 
                    if v['student'] == 'Обработка...'
                ]
                
                for idx, i in enumerate(violations_to_process):
                    face_status.text(f"Обработка нарушения {idx + 1}/{len(violations_to_process)}...")
                    violation_path = st.session_state.violations_log[i]['path']
                    
                    try:
                        name, score, face_path = st.session_state.face_recognizer.analyze_video_segment(
                            violation_path,
                            face_similarity=face_similarity
                        )
                        st.session_state.violations_log[i]['student'] = name
                        st.session_state.violations_log[i]['confidence'] = f"{score:.0%}"
                    except Exception as e:
                        st.error(f"Ошибка при анализе {Path(violation_path).name}: {str(e)}")
                        st.session_state.violations_log[i]['student'] = "Не опознан"
                        st.session_state.violations_log[i]['confidence'] = "Ошибка анализа"
                    
                    progress_face.progress((idx + 1) / len(violations_to_process))
                
                face_status.empty()
                progress_face.empty()
            else:
                for i, violation in enumerate(st.session_state.violations_log):
                    if violation['student'] == 'Обработка...':
                        st.session_state.violations_log[i]['student'] = "Не опознан"
                        st.session_state.violations_log[i]['confidence'] = "Нет БД"
        
        st.success(f"Обработка завершена! Обнаружено {len(st.session_state.violations_log)} нарушений.")
        st.session_state.processing = False
    
    except Exception as e:
        st.error(f"Ошибка при обработке видео: {e}")
        st.session_state.processing = False

def process_video_url(url, frame_skip=2, buffer_seconds=10, sleep_buffer=10, face_db_path="students.pkl", face_similarity=0.5):
    """Обработка видеопотока с URL"""
    try:
        st.info(f"Подключение к потоку: {url}")
        
        cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            st.error("Ошибка подключения к потоку!")
            st.info("""
            **Возможные причины:**
            - Неправильный URL (проверьте формат)
            - Потоковый сервис недоступен или требует аутентификации
            - Сеть недоступна или слабое соединение
            
            **Поддерживаемые форматы URL:**
            - RTSP потоки: `rtsp://...`
            - HTTP потоки: `http://... или https://...`
            - Файловые потоки: `/path/to/video.mp4`
            - Номер камеры: `0` (веб-камера)
            
            **Примеры:**
            - `rtsp://admin:password@192.168.1.100:554/stream`
            - `http://example.com/video.m3u8`
            """)
            st.session_state.processing = False
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 30
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if width == 0:
            width = 1280
        if height == 0:
            height = 720
        
        st.success(f"Подключено! Разрешение: {width}x{height}@{fps}fps")
        
        st.session_state.video_processor.setup_output_dirs()
        
        metrics = {
            'total_frames': 0,
            'violations': {},
            'recording': False,
            'frames_processed': 0
        }
        
        frame_count = 0
        sleep_start_time = None
        last_detection_time = None
        last_recording_end_time = None
        recording = False
        rec_violations = set()
        current_segment_path = None
        last_detections = {}
        last_confirmed_violations = set()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        frame_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
        metrics_placeholder = st.empty()
        
        st.info("Обработка видеопотока...")
        max_frames = 3000
        
        while cap.isOpened() and frame_count < max_frames and st.session_state.processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            metrics['total_frames'] = frame_count
            
            annotated_frame = frame.copy()
            current_time = time.time()
            confirmed_violations = set()
            
            if frame_count % frame_skip == 0:
                detections, _ = st.session_state.detector.detect_frame(frame, draw_boxes=False)
                last_detections = detections
                
                for class_name in detections:
                    if class_name not in metrics['violations']:
                        metrics['violations'][class_name] = 0
                    metrics['violations'][class_name] += 1
                
                confirmed_violations, sleep_start_time, detection_time = process_frame_for_detection(
                    current_time, set(detections.keys()), sleep_start_time, sleep_buffer
                )
                last_confirmed_violations = confirmed_violations
                if confirmed_violations and detection_time:
                    last_detection_time = detection_time
                
                if confirmed_violations and not recording:
                    segments_dir, _ = st.session_state.video_processor.setup_output_dirs()
                    filename = st.session_state.video_processor.generate_segment_filename()
                    current_segment_path = os.path.join(segments_dir, filename)
                    
                    st.session_state.video_processor.start_recording(
                        current_segment_path,
                        (width, height),
                        fps
                    )
                    recording = True
                    rec_violations = set(confirmed_violations)
                
                if confirmed_violations and recording:
                    rec_violations.update(confirmed_violations)
            
            if last_detections and last_confirmed_violations:
                violations_to_draw = last_confirmed_violations & set(last_detections.keys())
                if violations_to_draw:
                    annotated_frame = st.session_state.detector.draw_detections(
                        annotated_frame, last_detections, violations_to_draw
                    )
            
            if recording:
                cv2.circle(annotated_frame, (30, 30), 10, (0, 0, 255), -1)
                st.session_state.video_processor.write_frame(annotated_frame)
            
            if recording and last_detection_time:
                if (current_time - last_detection_time) > buffer_seconds:
                    st.session_state.video_processor.stop_recording()
                    recording = False
                    last_recording_end_time = current_time
                    
                    st.session_state.violations_log.append({
                        'path': current_segment_path,
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'violation': ", ".join(rec_violations),
                        'student': 'Обработка...',
                        'confidence': 'N/A'
                    })
            
            frame_placeholder.image(annotated_frame, channels="BGR")
            metrics_placeholder.write(f"Обработано кадров: {metrics['total_frames']} | Нарушений: {len(st.session_state.violations_log)}")
            progress = min(frame_count / max_frames, 1.0)
            progress_bar.progress(progress)
        
        cap.release()
        if recording:
            st.session_state.video_processor.stop_recording()
        
        if st.session_state.violations_log:
            st.info("Анализ лиц в обнаруженных нарушениях...")
            
            if st.session_state.face_recognizer is None and os.path.exists(face_db_path):
                st.session_state.face_recognizer = load_face_recognizer(face_db_path)
            
            if st.session_state.face_recognizer and st.session_state.face_recognizer.is_database_available():
                progress_face = st.progress(0)
                face_status = st.empty()
                
                violations_to_process = [
                    i for i, v in enumerate(st.session_state.violations_log) 
                    if v['student'] == 'Обработка...'
                ]
                
                for idx, i in enumerate(violations_to_process):
                    face_status.text(f"Обработка нарушения {idx + 1}/{len(violations_to_process)}...")
                    violation_path = st.session_state.violations_log[i]['path']
                    
                    try:
                        name, score, face_path = st.session_state.face_recognizer.analyze_video_segment(
                            violation_path,
                            face_similarity=face_similarity
                        )
                        st.session_state.violations_log[i]['student'] = name
                        st.session_state.violations_log[i]['confidence'] = f"{score:.0%}"
                    except Exception as e:
                        st.error(f"Ошибка при анализе {Path(violation_path).name}: {str(e)}")
                        st.session_state.violations_log[i]['student'] = "Не опознан"
                        st.session_state.violations_log[i]['confidence'] = "Ошибка анализа"
                    
                    progress_face.progress((idx + 1) / len(violations_to_process))
                
                face_status.empty()
                progress_face.empty()
            else:
                for i, violation in enumerate(st.session_state.violations_log):
                    if violation['student'] == 'Обработка...':
                        st.session_state.violations_log[i]['student'] = "Не опознан"
                        st.session_state.violations_log[i]['confidence'] = "Нет БД"
        
        st.success(f"Обработка завершена! Обнаружено {len(st.session_state.violations_log)} нарушений.")
        st.session_state.processing = False
    
    except Exception as e:
        st.error(f"Ошибка при обработке потока: {e}")
        st.session_state.processing = False

def process_violations_data(violations_log):
    """Обработка данных нарушений для анализа"""
    import pandas as pd
    return pd.DataFrame(violations_log) if violations_log else None

def generate_report(violations_log, face_db_path):
    """Генерирует текстовый отчет"""
    try:
        if not st.session_state.face_recognizer and os.path.exists(face_db_path):
            st.session_state.face_recognizer = load_face_recognizer(face_db_path)
        
        return st.session_state.video_processor.generate_report(
            violations_log,
            st.session_state.face_recognizer
        )
    except Exception as e:
        st.error(f"Ошибка при генерации отчета: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ ИНТЕРФЕЙС
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">Система Мониторинга Дисциплины</div>', 
            unsafe_allow_html=True)

# Боковая панель с настройками
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Настройки</div>', unsafe_allow_html=True)
    
    # Автоматическая загрузка модели best.pt
    model_path = "best.pt"
    if st.session_state.detector is None and os.path.exists(model_path):
        st.session_state.detector = load_detector(model_path)
    
    with st.expander("🎥 Детекция", expanded=True):
        conf_threshold = st.slider(
            "Порог уверенности",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Чем выше - тем строже критерии"
        )
        frame_skip = st.slider(
            "Пропуск кадров",
            min_value=1,
            max_value=10,
            value=2,
            help="Обрабатывать каждый N-й кадр (для скорости)"
        )
    
    with st.expander("⏱ Запись", expanded=True):
        buffer_seconds = st.slider(
            "Буфер после нарушения (сек)",
            min_value=5,
            max_value=30,
            value=10,
            help="Сколько секунд писать после исчезновения нарушения"
        )
        sleep_buffer = st.slider(
            "Буфер подтверждения сна (сек)",
            min_value=5,
            max_value=30,
            value=10,
            help="Таймер подтверждения сна перед записью"
        )
    
    with st.expander("👤 Распознавание лиц", expanded=True):
        face_db_path = "students.pkl"
        face_similarity = st.slider(
            "Порог сходства лица",
            min_value=0.3,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Минимальное сходство для опознания"
        )
    
    st.divider()
    model_status = "✅ загружена" if st.session_state.detector else "❌ не найдена"
    st.caption(f"Модель: {model_status}")
    
    # Индикатор состояния обработки
    if st.session_state.processing:
        st.info(f"🔄 Обработка... Нарушений: {len(st.session_state.violations_log)}")
    else:
        if st.session_state.violations_log:
            st.success(f"✅ Готово. Нарушений: {len(st.session_state.violations_log)}")

if st.session_state.detector is None:
    st.warning("Модель не загружена. Проверьте наличие файла best.pt")
else:
    # Основные вкладки
    tab1, tab2 = st.tabs([
        "Обработка видео",
        "Статистика"
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # ВКЛАДКА 1: ОБРАБОТКА ВИДЕО
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab1:
        st.header("Обработка видео")
        
        col_source, _ = st.columns([2, 1])  # вторая колонка не нужна, кнопки под радио
        
        with col_source:
            video_source = st.radio(
                "Выберите источник видео:",
                ["Веб-камера", "Видеофайл", "URL потока"],
                horizontal=True
            )
            
            # Кнопки управления
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                start_disabled = st.session_state.processing
                if st.button("▶ Начать обработку", key="process_btn", disabled=start_disabled, use_container_width=True):
                    if not st.session_state.processing:
                        st.session_state.processing = True
                        st.rerun()
            with btn_col2:
                if st.session_state.processing:
                    if st.button("⏹ Остановить", key="stop_btn", use_container_width=True):
                        st.session_state.processing = False
                        st.rerun()
        
        # Контейнеры для видео и метрик
        video_container = st.container()
        metrics_container = st.container()
        
        if st.session_state.processing:
            if video_source == "Веб-камера":
                process_webcam(frame_skip, buffer_seconds, sleep_buffer,
                             face_db_path, face_similarity)
            
            elif video_source == "Видеофайл":
                video_file = st.file_uploader(
                    "Загрузите видеофайл",
                    type=['mp4', 'avi', 'mov', 'mkv']
                )
                if video_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                        tmp.write(video_file.read())
                        process_video_file(tmp.name, frame_skip, buffer_seconds, sleep_buffer, 
                                         face_db_path, face_similarity)
            
            elif video_source == "URL потока":
                url = st.text_input("Введите URL видеопотока:")
                if url:
                    process_video_url(url, frame_skip, buffer_seconds, sleep_buffer,
                                     face_db_path, face_similarity)
    
    # ═══════════════════════════════════════════════════════════════════════
    # ВКЛАДКА 2: СТАТИСТИКА И ЖУРНАЛ
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab2:
        st.header("Статистика нарушений")
        
        if st.session_state.violations_log:
            violations_df = process_violations_data(st.session_state.violations_log)
            
            # Счетчики
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(st.session_state.violations_log)}</div>
                    <div class="metric-label">Всего нарушений</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                sleeping_count = sum(1 for v in st.session_state.violations_log if 'sleeping' in v['violation'].lower())
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{sleeping_count}</div>
                    <div class="metric-label">Сон</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                phone_count = sum(1 for v in st.session_state.violations_log if 'phone' in v['violation'].lower())
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{phone_count}</div>
                    <div class="metric-label">Телефон</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                food_count = sum(1 for v in st.session_state.violations_log 
                                if 'food' in v['violation'].lower() or 'bottle' in v['violation'].lower())
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{food_count}</div>
                    <div class="metric-label">Еда/Напиток</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Графики
            st.subheader("Распределение по типам нарушений")
            violation_types = {}
            for v in st.session_state.violations_log:
                for vtype in v['violation'].split(', '):
                    violation_types[vtype] = violation_types.get(vtype, 0) + 1
            
            if violation_types:
                import plotly.express as px
                fig = px.bar(
                    x=list(violation_types.keys()),
                    y=list(violation_types.values()),
                    labels={'x': 'Тип нарушения', 'y': 'Количество'},
                    color=['#e74c3c', '#e67e22', '#3498db', '#9b59b6'][:len(violation_types)]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Временная шкала
            st.subheader("Временная шкала нарушений")
            times = [v['time'] for v in st.session_state.violations_log]
            st.write(f"Первое нарушение: {times[0] if times else 'N/A'}")
            st.write(f"Последнее нарушение: {times[-1] if times else 'N/A'}")
            st.write(f"Всего записано: {len(st.session_state.violations_log)} фрагментов")
            
            st.divider()
            
            # Заголовок журнала с кнопками экспорта/очистки
            col_title, col_export, col_clear = st.columns([3, 1, 1])
            with col_title:
                st.subheader("📋 Журнал нарушений")
            with col_export:
                import pandas as pd
                df = pd.DataFrame(st.session_state.violations_log)
                csv = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8')
                st.download_button(
                    label="📥 CSV",
                    data=csv,
                    file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="csv_download_tab2"
                )
            with col_clear:
                if st.button("🗑 Очистить", disabled=st.session_state.processing, use_container_width=True):
                    st.session_state.violations_log = []
                    st.rerun()
            
            st.divider()
            st.subheader("Отчет")

            if st.button("Сгенерировать отчет", key="gen_report", use_container_width=True):
                report_path = generate_report(st.session_state.violations_log, face_db_path)
                if report_path and os.path.exists(report_path):
                    st.success(f"Отчет создан: {report_path}")

                    with open(report_path, "r", encoding="utf-8") as f:
                        st.text(f.read())

                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="Скачать отчет",
                            data=f.read(),
                            file_name=Path(report_path).name,
                            mime="text/plain",
                            key="download_report"
                        )
            
            for i, violation in enumerate(st.session_state.violations_log, 1):
                with st.expander(
                    f"Нарушение #{i} | {violation['time']} | {violation['violation']}"
                ):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write(f"**Время:** {violation['time']}")
                        st.write(f"**Нарушение:** {violation['violation']}")
                        st.write(f"**Файл:** {Path(violation['path']).name}")
                    
                    with col2:
                        st.write(f"**Студент:** {violation.get('student', 'Неизвестно')}")
                        st.write(f"**Уверенность:** {violation.get('confidence', 'N/A')}")
                    
                    if os.path.exists(violation['path']):
                        with open(violation['path'], 'rb') as f:
                            st.download_button(
                                label="Скачать видео",
                                data=f.read(),
                                file_name=Path(violation['path']).name,
                                mime="video/mp4",
                                key=f"download_{i}"
                            )
        
        else:
            st.info("Нет данных для отображения. Обработайте видео сначала.")

if __name__ == "__main__":
    pass