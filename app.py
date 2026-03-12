from flask import Flask, render_template
from predictor import predict_bp
from historical import historical_bp
from correlation import correlation_bp
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "default-secret-key-change-in-production")

# Make Python's zip available in all Jinja2 templates
app.jinja_env.globals.update(zip=zip)

# Register Blueprints
app.register_blueprint(predict_bp)
app.register_blueprint(historical_bp)
app.register_blueprint(correlation_bp)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/health")
def health():
    return {"status": "healthy", "message": "Stock Predictor API is running"}

@app.errorhandler(404)
def not_found(e):
    return render_template("home.html", error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("home.html", error="Internal server error. Please try again."), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)