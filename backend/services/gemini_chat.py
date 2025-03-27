
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize model
model = genai.GenerativeModel("gemini-1.5-pro")

def get_mood_recommendation(mood):
    recommendations = {
        "Neutral": "Let's do something fun! Would you like to play a quick game, hear an interesting fact, or get a fun challenge?",
        "Happy": "You're in a great mood! Let's keep the positivity going. Want to listen to an uplifting story, get a fun challenge, or share a happy memory?",
        "Sad": "I'm here for you. Would you like some calming music recommendations, a motivational story, or a virtual hug?",
        "Angry": "I understand frustration can be tough. Want to try a quick breathing exercise, hear a calming story, or vent your thoughts?",
        "Fearful": "You're not alone! Let's ease your mind. Want a comforting quote, a guided relaxation, or an inspiring success story?",
        "Disgusted": "Let's distract your mind. Want to hear a joke, learn a new interesting fact, or play a quick word game?",
        "Surprised": "Surprises can be fun! Want to share what happened, hear about a surprising event, or play a quick reaction game?",
        "Calm": "A peaceful moment is precious. Would you like to try meditation, listen to soothing nature sounds, or explore a new mindfulness tip?"
    }
    return recommendations.get(mood, "I didn't recognize that mood. Can you describe how you're feeling?")

def process_message(message):
    # Check if message matches an emotion
    if message.capitalize() in ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised", "Calm"]:
        return get_mood_recommendation(message.capitalize())
    else:
        # Use Gemini for other messages
        response = model.generate_content(message)
        return response.text
