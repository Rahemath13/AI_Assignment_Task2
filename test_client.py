import requests

payload = {"ingredients": "egg, onion", "num_return_sequences": 1, "max_length": 60}
try:
    resp = requests.post("http://127.0.0.1:8000/generate", json=payload, timeout=60)
    print("Status:", resp.status_code)
    print("Response:", resp.text)
except Exception as e:
    print("Error:", e)
