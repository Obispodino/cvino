import streamlit as st

st.title("Manual audio test")

st.components.v1.html("""
<audio controls>
  <source src="https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3" type="audio/mp3">
</audio>
""", height=60)
