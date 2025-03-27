
from flask import Flask, request, jsonify
from flask_cors import CORS
from services.emotion_detector import process_video
from services.audio_emotion_detector import process_audio
from services.gemini_chat import process_message
from services.google_fit_service import fetch_wellness_metrics
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/wellness-metrics', methods=['GET'])
def get_wellness_metrics():
    try:
        logger.info("Fetching wellness metrics...")
        metrics = fetch_wellness_metrics()
        logger.info(f"Metrics fetched: {metrics}")
        if metrics:
            return jsonify(metrics)
        return jsonify({'error': 'Unable to fetch wellness metrics'}), 500
    except Exception as e:
        logger.error(f"Error fetching wellness metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500
            
@app.route('/api/detect-emotion', methods=['POST'])
def detect_emotion():
    if 'video' in request.files:
        video_file = request.files['video']
        try:
            logger.info(f"Processing video: {video_file.filename}")
            emotion_results = process_video(video_file)
            
            # Get Gemini response for the detected emotion
            emotion_response = process_message(emotion_results['dominant_emotion'])
            emotion_results['response'] = emotion_response
            
            logger.info(f"Video processing complete: {emotion_results}")
            return jsonify(emotion_results)
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return jsonify({'error': str(e)}), 500
            
    elif 'audio' in request.files:
        audio_file = request.files['audio']
        try:
            logger.info(f"Processing audio: {audio_file.filename}")
            emotion_results = process_audio(audio_file)
            
            # Get Gemini response for the detected emotion
            emotion_response = process_message(emotion_results['dominant_emotion'])
            emotion_results['response'] = emotion_response
            
            logger.info(f"Audio processing complete: {emotion_results}")
            return jsonify(emotion_results)
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'No video or audio file provided'}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message')
        if not message:
            return jsonify({'error': 'No message provided'}), 400
            
        response = process_message(message)
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True, port=5000, host='0.0.0.0')
