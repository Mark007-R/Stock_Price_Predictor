from flask import Flask, render_template
from predictor import predict_bp
from historical import historical_bp
from correlation import correlation_bp
import os
load_dotenv() 
app = Flask(__name__)

# Secret key for session management
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY")

# Register Blueprints
app.register_blueprint(predict_bp)
app.register_blueprint(historical_bp)
app.register_blueprint(correlation_bp)

# Home route
@app.route("/")
def home():
    return render_template("home.html")

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template("home.html", error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("home.html", error="Internal server error. Please try again."), 500

if __name__ == "__main__":
    app.run(debug=True)