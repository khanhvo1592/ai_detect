import os
import cv2
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import pytesseract

# Thiết lập logging
log_file = 'server.log'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Thiết lập đường dẫn TESSDATA_PREFIX
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata/'

# Chỉ định đường dẫn Tesseract (nếu cần thiết)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Đối với Linux
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Đối với Windows

def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='vie')
    return text.strip()

@app.route('/upload_chunk', methods=['POST'])
def upload_chunk():
    try:
        upload_id = request.form['uploadId']
        chunk_index = int(request.form['chunkIndex'])
        chunk = request.files['file']
        
        upload_path = os.path.join(UPLOAD_FOLDER, f"{upload_id}_{chunk_index}")
        chunk.save(upload_path)
        
        logging.info(f"Chunk {chunk_index} uploaded successfully for upload ID {upload_id}")
        
        return jsonify({'message': 'Chunk uploaded successfully'}), 200
    except Exception as e:
        logging.error(f"Error uploading chunk: {e}")
        return jsonify({'message': 'Error uploading chunk', 'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        upload_id = request.json['uploadId']
        
        # Combine chunks
        combined_path = os.path.join(UPLOAD_FOLDER, f"{upload_id}_combined.mp4")
        with open(combined_path, 'wb') as combined_file:
            chunk_index = 0
            while True:
                chunk_path = os.path.join(UPLOAD_FOLDER, f"{upload_id}_{chunk_index}")
                if not os.path.exists(chunk_path):
                    break
                with open(chunk_path, 'rb') as chunk_file:
                    combined_file.write(chunk_file.read())
                os.remove(chunk_path)
                chunk_index += 1
        
        logging.info(f"Chunks combined successfully for upload ID {upload_id}")
        
        # Analyze the combined video
        analysis_result = analyze_video(combined_path)
        
        return jsonify(analysis_result), 200
    except Exception as e:
        logging.error(f"Error analyzing video: {e}")
        return jsonify({'message': 'Error analyzing video', 'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        # Load history from a JSON file
        history_path = os.path.join(UPLOAD_FOLDER, 'history.json')
        if not os.path.exists(history_path):
            return jsonify([]), 200

        with open(history_path, 'r') as history_file:
            history = json.load(history_file)

        return jsonify(history), 200
    except Exception as e:
        logging.error(f"Error fetching history: {e}")
        return jsonify({'message': 'Error fetching history', 'error': str(e)}), 500

def analyze_video(video_path):
    try:
        # Load YOLOv8 model
        model_path = 'models/bantin_hgtv.pt'
        model = YOLO(model_path)
        logging.info(f"YOLO model loaded from {model_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error('Error opening video file')
            return {'error': 'Error opening video file'}
        logging.info(f"Video file opened: {video_path}")

        frame_count = 0
        results_per_frame = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps * 10)  # Number of frames to skip (10 seconds interval)
        logging.info(f"Processing video at {fps} FPS, skipping every {frame_skip} frames")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 'frame_skip' frame
            if frame_count % frame_skip == 0:
                # Perform detection
                results = model(frame)
                logging.info(f"Processed frame {frame_count}")

                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
                    scores = result.boxes.conf.cpu().numpy()  # Get confidence scores
                    class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs

                    # Filter results with confidence > 0.9
                    for box, score, class_id in zip(boxes, scores, class_ids):
                        if score >= 0.9:
                            x1, y1, x2, y2 = map(int, box)
                            roi = frame[y1:y2, x1:x2]  # Extract region of interest
                            text = extract_text_from_image(roi)  # Extract text from the ROI

                            if text:  # Only save results with text
                                timecode = frame_count / fps
                                minutes = int(timecode // 60)
                                seconds = int(timecode % 60)
                                formatted_timecode = f"{minutes:02}:{seconds:02}"
                                result_dict = {
                                    "timecode": formatted_timecode,
                                    "text": text
                                }
                                results_per_frame.append(result_dict)

            frame_count += 1

        cap.release()

        # Remove results with text length < 10 characters and duplicate timecodes
        unique_results = {}
        for result in results_per_frame:
            if len(result['text']) >= 10:
                if result['timecode'] not in unique_results or len(unique_results[result['timecode']]['text']) < len(result['text']):
                    unique_results[result['timecode']] = result

        logging.info(f"Analysis completed for video: {video_path}")
        
        # Save analysis result to history
        history_path = os.path.join(UPLOAD_FOLDER, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as history_file:
                history = json.load(history_file)
        else:
            history = []

        history.append({
            "video_path": video_path,
            "results": list(unique_results.values())
        })

        with open(history_path, 'w') as history_file:
            json.dump(history, history_file, indent=4, ensure_ascii=False)

        return list(unique_results.values())
    except Exception as e:
        logging.error(f"Error analyzing video: {e}")
        return {'error': 'Error analyzing video', 'message': str(e)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
