import io


def listen_voice(audio_file):
    """
    Convert browser-recorded audio into text.

    The audio is captured by Streamlit's st.audio_input()
    in the user's browser and then processed using
    SpeechRecognition + Google Speech Recognition API.
    """

    try:
        import speech_recognition as sr
    except ImportError:
        return {
            "success": False,
            "text": "",
            "error": (
                "SpeechRecognition library not installed. "
                "Add SpeechRecognition to requirements.txt."
            )
        }

    if audio_file is None:
        return {
            "success": False,
            "text": "",
            "error": "No audio recorded. Please record your voice and try again."
        }

    recognizer = sr.Recognizer()

    try:
        # Get audio bytes recorded by Streamlit
        audio_bytes = audio_file.getvalue()

        if not audio_bytes:
            return {
                "success": False,
                "text": "",
                "error": "The recorded audio is empty. Please try again."
            }

        # Convert bytes into an in-memory file
        audio_buffer = io.BytesIO(audio_bytes)

        # Read the browser-recorded WAV audio
        with sr.AudioFile(audio_buffer) as source:
            audio = recognizer.record(source)

        # Convert speech to text using Google Speech Recognition
        text = recognizer.recognize_google(audio)

        return {
            "success": True,
            "text": text.strip(),
            "error": ""
        }

    except sr.UnknownValueError:
        return {
            "success": False,
            "text": "",
            "error": (
                "Could not understand your speech. "
                "Please speak clearly and try again."
            )
        }

    except sr.RequestError:
        return {
            "success": False,
            "text": "",
            "error": (
                "Internet connection is required for voice recognition. "
                "Please check your connection and try again."
            )
        }

    except ValueError:
        return {
            "success": False,
            "text": "",
            "error": (
                "The recorded audio format could not be processed. "
                "Please record again."
            )
        }

    except Exception as e:
        return {
            "success": False,
            "text": "",
            "error": f"Voice recognition error: {str(e)}"
        }


def process_voice_result(result):
    """
    Process the result returned by listen_voice().
    """

    if not result["success"]:
        return None, result["error"]

    return result["text"], None
