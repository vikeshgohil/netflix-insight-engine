import streamlit as st

def listen_voice():
    try:
        import speech_recognition as sr
    except ImportError:
        return {
            'success': False,
            'text': '',
            'error': 'SpeechRecognition library not installed. Run: pip install SpeechRecognition'
        }

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)

        text = recognizer.recognize_google(audio)
        return {
            'success': True,
            'text': text.strip(),
            'error': ''
        }

    except sr.WaitTimeoutError:
        return {
            'success': False,
            'text': '',
            'error': 'No speech detected. Please click the button and speak clearly.'
        }

    except sr.UnknownValueError:
        return {
            'success': False,
            'text': '',
            'error': 'Could not understand your speech. Please try again in a quieter environment.'
        }

    except sr.RequestError:
        return {
            'success': False,
            'text': '',
            'error': 'Internet connection required for voice recognition. Please check your connection.'
        }

    except OSError:
        return {
            'success': False,
            'text': '',
            'error': 'Microphone not found. Please check your microphone is connected and working.'
        }

    except Exception as e:
        return {
            'success': False,
            'text': '',
            'error': f'Voice recognition error: {str(e)}'
        }


def process_voice_result(result):
    if not result['success']:
        return None, result['error']
    return result['text'], None


def is_microphone_available():
    try:
        import speech_recognition as sr
        import pyaudio
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        p.terminate()
        return device_count > 0
    except Exception:
        return False