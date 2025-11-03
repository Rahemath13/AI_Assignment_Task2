# 🍳 AI Recipe Chatbot

An AI-powered recipe generator that suggests cooking recipes based on the ingredients you enter.  
This project fine-tunes GPT-2 on a small recipe dataset and provides a **FastAPI backend** with a **Streamlit frontend** for easy interaction.

---

## 🧰 Requirements

- Python 3.10 or later
- Works on **Windows / Linux / macOS**
- No external setup (just follow steps below)

---

## ⚙️ Setup Instructions

### 1️⃣ Extract the Project
Unzip the folder `AI_Recipe_Chatbot.zip` anywhere on your system.

Open a terminal in that folder.

---

### 2️⃣ Create and Activate a Virtual Environment

#### 🪟 On Windows
```bash
python -m venv .venv
.venv\Scripts\activate

🐧 On Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the FastAPI Backend

Open Terminal 1 and start the API server:

uvicorn serve_api:app --reload


✅ You should see output like:

✅ Generator pipeline ready (device=cpu)
INFO:     Application startup complete.

5️⃣ Run the Streamlit Frontend

Open another terminal (keep backend running):

streamlit run chatbot_ui.py


✅ Your browser will automatically open at:

http://localhost:8501

🧠 How to Use

Type ingredients like:

egg, onion


Click Get Recipe

✅ Example Output
🥘 Recipe 1
🍽️ Quick Onion Omelette

1. Beat 2 eggs.
2. Chop onion finely.
3. Sauté onion until golden.
4. Add eggs and cook until set.

