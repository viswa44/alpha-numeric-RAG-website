from pydub import AudioSegment
import streamlit as st
import speech_recognition as sr

# Convert audio to WAV format
def convert_audio_to_wav(audio_file):
    try:
        audio = AudioSegment.from_file(audio_file)
        converted_file = "converted_audio.wav"
        audio.export(converted_file, format="wav")
        return converted_file
    except Exception as e:
        st.error(f"Error converting audio file: {e}")
        return None

# Transcribe audio
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        # Convert to WAV if necessary
        converted_file = convert_audio_to_wav(audio_file)
        if not converted_file:
            return None

        # Transcribe the converted audio
        with sr.AudioFile(converted_file) as source:
            st.info("Processing the audio file for transcription...")
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            st.success(f"Transcribed Text: {text}")
            return text
    except sr.UnknownValueError:
        st.error("Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Error with Speech Recognition service: {e}")
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
    return None

# Streamlit UI
st.title("Transcribe Uploaded Audio")

# File uploader for pre-recorded audio
audio_file = st.file_uploader("Upload an audio file for transcription (e.g., WAV, MP3)", type=["wav", "mp3", "flac"])

if st.button("🎙️ Transcribe Audio"):
    if audio_file:
        transcription = transcribe_audio(audio_file)
        if transcription:
            st.text_area("Transcribed Text", transcription)
    else:
        st.warning("Please upload an audio file.")
