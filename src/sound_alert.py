import base64
import streamlit as st
import os

def play_drowsy_sound():
    """
    Plays alarm sound when drowsiness detected.
    """
    sound_path = os.path.join(os.path.dirname(__file__), "alarm.wav")

    if not os.path.exists(sound_path):
        st.warning("Alarm sound file not found.")
        return

    with open(sound_path, "rb") as f:
        data = f.read()

    b64 = base64.b64encode(data).decode()

    sound_html = f"""
    <audio id="alarm" autoplay loop>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    <script>
        setTimeout(function() {{
            var audio = document.getElementById('alarm');
            audio.pause();
        }}, 30000);
    </script>
    """

    st.markdown(sound_html, unsafe_allow_html=True)