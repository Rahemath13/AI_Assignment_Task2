# chatbot_ui.py
import streamlit as st
import requests

st.set_page_config(page_title="Local Recipe Chatbot", page_icon="🍳", layout="centered")
API_URL = "http://127.0.0.1:8000/generate"

st.title("🍳 Local Recipe Chatbot")
st.write("Enter ingredients (comma-separated) to get recipe ideas!")

ingredients = st.text_input("Ingredients", value="egg, onion")
# Fixed values (no sliders shown)
num_return_sequences = 1
max_length = 80

if st.button("Get Recipe"):
    if ingredients.strip():
        payload = {"ingredients": ingredients, "num_return_sequences": 1, "max_length": 80}
        with st.spinner("Generating recipe... please wait ⏳"):
            try:
                resp = requests.post(API_URL, json=payload, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    if "recipes" in data:
                        for i, recipe in enumerate(data["recipes"], start=1):
                            st.markdown(f"### 🥘 Recipe {i}")
                            st.write(recipe)
                            st.divider()
                    else:
                        st.error(f"API response error: {data}")
                else:
                    st.error(f"Error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
    else:
        st.warning("Please enter ingredients before clicking Get Recipe.")
