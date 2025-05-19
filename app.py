from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from React frontend

@app.route("/", methods=["GET"])
def index():
    return "Server is running!", 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip().lower()

    if message == "hi":
        response = "hello"
    else:
        response = "I don't understand."

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
