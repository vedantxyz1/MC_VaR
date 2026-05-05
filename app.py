from flask import Flask, jsonify, render_template
import var_engine

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/data")
def data():
    try:
        return jsonify(var_engine.run())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
