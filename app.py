import os
import pickle
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)
from flask_bcrypt import Bcrypt

from db import db

load_dotenv()

app = Flask(__name__)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "cropwise_default_secret_key")



DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
else:
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    app.config["SQLALCHEMY_DATABASE_URI"] = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message_category = "danger"



class User(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    histories = db.relationship(
        "PredictionHistory",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan"
    )


class PredictionHistory(db.Model):
    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)

    predicted_crop = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



MODEL_PATH = os.path.join(app.root_path, "RandomForest.pkl")

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
    print("ERROR: RandomForest.pkl was not found. Make sure it is in the project root.")



CROP_INFO = {
    "rice": {
        "image": "rice.jpg",
        "reason": "Rice grows well in warm temperatures, high humidity, and areas with sufficient rainfall. It is suitable for soil conditions that can hold water.",
        "fertilizer": "Use nitrogen-rich fertilizer such as urea, and apply phosphorus and potassium based on soil test results.",
        "tips": "Maintain proper water levels in the field and avoid over-flooding during early growth stages."
    },
    "maize": {
        "image": "maize.jpg",
        "reason": "Maize is suitable for moderately warm temperatures, balanced nutrients, and well-drained soil.",
        "fertilizer": "Apply nitrogen fertilizer during early growth and add phosphorus for strong root development.",
        "tips": "Plant in rows with proper spacing and ensure enough sunlight exposure."
    },
    "chickpea": {
        "image": "chickpea.jpg",
        "reason": "Chickpea performs well in dry and cool conditions with moderate soil nutrients.",
        "fertilizer": "Use phosphorus-based fertilizer and avoid too much nitrogen because chickpea can fix nitrogen naturally.",
        "tips": "Avoid waterlogging and plant in well-drained soil."
    },
    "kidneybeans": {
        "image": "kidneybeans.jpg",
        "reason": "Kidney beans grow best in fertile, well-drained soil with moderate temperature and moisture.",
        "fertilizer": "Use compost and phosphorus-rich fertilizer to support pod formation.",
        "tips": "Avoid excessive watering and provide support if needed."
    },
    "pigeonpeas": {
        "image": "pigeonpeas.jpg",
        "reason": "Pigeon peas tolerate dry conditions and grow well in moderate soil fertility.",
        "fertilizer": "Apply phosphorus fertilizer and organic compost before planting.",
        "tips": "Plant in sunny areas and avoid heavy waterlogged soil."
    },
    "mothbeans": {
        "image": "mothbeans.jpg",
        "reason": "Moth beans are drought-tolerant and suitable for dry soil conditions.",
        "fertilizer": "Use organic manure and small amounts of phosphorus fertilizer.",
        "tips": "Best grown in warm climates with minimal irrigation."
    },
    "mungbean": {
        "image": "mungbean.jpg",
        "reason": "Mungbean grows well in warm temperatures with moderate rainfall and well-drained soil.",
        "fertilizer": "Apply phosphorus and potassium fertilizer. Avoid too much nitrogen.",
        "tips": "Use good-quality seeds and avoid planting during heavy rain."
    },
    "blackgram": {
        "image": "blackgram.jpg",
        "reason": "Blackgram is suitable for warm climates and moderate soil moisture.",
        "fertilizer": "Use compost and phosphorus fertilizer for better root and pod growth.",
        "tips": "Avoid waterlogging and control weeds during early growth."
    },
    "lentil": {
        "image": "lentil.jpg",
        "reason": "Lentil grows best in cool weather and well-drained soil with moderate fertility.",
        "fertilizer": "Apply phosphorus fertilizer and organic matter.",
        "tips": "Do not over-irrigate because lentils are sensitive to excess water."
    },
    "pomegranate": {
        "image": "pomegranate.jpg",
        "reason": "Pomegranate is suitable for warm and dry climates with well-drained soil.",
        "fertilizer": "Use compost, potassium, and balanced NPK fertilizer.",
        "tips": "Prune regularly and avoid excessive irrigation."
    },
    "banana": {
        "image": "banana.jpg",
        "reason": "Banana needs high humidity, warm temperature, and nutrient-rich soil.",
        "fertilizer": "Apply potassium-rich fertilizer, compost, and nitrogen during vegetative growth.",
        "tips": "Keep soil moist and protect plants from strong winds."
    },
    "mango": {
        "image": "mango.jpg",
        "reason": "Mango grows well in warm climates with moderate rainfall and well-drained soil.",
        "fertilizer": "Apply organic compost and balanced NPK fertilizer.",
        "tips": "Avoid waterlogging and prune trees to improve fruit production."
    },
    "grapes": {
        "image": "grapes.jpg",
        "reason": "Grapes prefer warm temperatures, good sunlight, and well-drained soil.",
        "fertilizer": "Use potassium and phosphorus fertilizer during flowering and fruiting.",
        "tips": "Provide trellis support and prune vines regularly."
    },
    "watermelon": {
        "image": "watermelon.jpg",
        "reason": "Watermelon grows well in warm soil, high sunlight, and moderate rainfall.",
        "fertilizer": "Use compost and potassium-rich fertilizer for fruit development.",
        "tips": "Provide wide spacing and avoid overhead watering."
    },
    "muskmelon": {
        "image": "muskmelon.jpg",
        "reason": "Muskmelon is suitable for warm and dry conditions with well-drained soil.",
        "fertilizer": "Apply compost and potassium fertilizer.",
        "tips": "Use mulching to conserve soil moisture."
    },
    "apple": {
        "image": "apple.jpg",
        "reason": "Apple grows best in cooler climates with balanced nutrients and proper soil moisture.",
        "fertilizer": "Use nitrogen, phosphorus, and potassium depending on tree age and soil test.",
        "tips": "Prune yearly and protect from pests and fungal diseases."
    },
    "orange": {
        "image": "orange.jpg",
        "reason": "Orange prefers warm climates, moderate rainfall, and slightly acidic to neutral soil.",
        "fertilizer": "Use citrus fertilizer with nitrogen, potassium, and micronutrients.",
        "tips": "Water regularly but avoid waterlogging."
    },
    "papaya": {
        "image": "papaya.jpg",
        "reason": "Papaya grows well in warm temperatures, high humidity, and fertile soil.",
        "fertilizer": "Apply compost and balanced NPK fertilizer regularly.",
        "tips": "Plant in sunny areas and protect from strong winds."
    },
    "coconut": {
        "image": "coconut.jpg",
        "reason": "Coconut is suitable for humid tropical areas with high rainfall and sandy or loamy soil.",
        "fertilizer": "Use potassium-rich fertilizer and organic manure.",
        "tips": "Maintain proper spacing and provide irrigation during dry months."
    },
    "cotton": {
        "image": "cotton.jpg",
        "reason": "Cotton grows well in warm climates with moderate rainfall and fertile soil.",
        "fertilizer": "Apply nitrogen, phosphorus, and potassium fertilizer based on soil needs.",
        "tips": "Control weeds and pests early to protect boll formation."
    },
    "jute": {
        "image": "jute.jpg",
        "reason": "Jute grows well in warm, humid conditions with high rainfall.",
        "fertilizer": "Use nitrogen and phosphorus fertilizer with organic compost.",
        "tips": "Keep the field moist and harvest at the right maturity stage."
    },
    "coffee": {
        "image": "coffee.jpg",
        "reason": "Coffee grows best in moderate temperatures, good rainfall, and slightly acidic soil.",
        "fertilizer": "Use organic compost, nitrogen, and potassium fertilizer.",
        "tips": "Provide partial shade and maintain proper pruning."
    }
}



def get_crop_image_url(image_name):
    image_path = f"images/crops/{image_name}"
    full_image_path = os.path.join(app.root_path, "static", image_path)

    if os.path.exists(full_image_path):
        return url_for("static", filename=image_path)

    return url_for("static", filename="images/crops/default.jpg")



@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not fullname or not email or not password:
            flash("Please fill in all fields.", "danger")
            return redirect(url_for("signup"))

        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            flash("Email already exists. Please login instead.", "danger")
            return redirect(url_for("signup"))

        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

        new_user = User(
            fullname=fullname,
            email=email,
            password=hashed_password
        )

        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if model is None:
        return jsonify({
            "error": "Model file was not found. Please upload RandomForest.pkl."
        }), 500

    try:
        data = request.get_json()

        nitrogen = float(data["nitrogen"])
        phosphorus = float(data["phosphorus"])
        potassium = float(data["potassium"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        ph = float(data["ph"])
        rainfall = float(data["rainfall"])

        input_data = np.array([[
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall
        ]])

        predicted_crop = model.predict(input_data)[0]

        history = PredictionHistory(
            user_id=current_user.id,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            temperature=temperature,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall,
            predicted_crop=predicted_crop
        )

        db.session.add(history)
        db.session.commit()

        crop_details = CROP_INFO.get(predicted_crop, {
            "image": "default.jpg",
            "reason": "This crop matches the provided soil and climate conditions.",
            "fertilizer": "Use balanced fertilizer based on soil testing.",
            "tips": "Monitor moisture, soil pH, and pest activity regularly."
        })

        image_url = get_crop_image_url(crop_details["image"])

        return jsonify({
            "crop": predicted_crop.title(),
            "image": image_url,
            "reason": crop_details["reason"],
            "fertilizer": crop_details["fertilizer"],
            "tips": crop_details["tips"]
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


@app.route("/history")
@login_required
def history():
    records = PredictionHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(PredictionHistory.created_at.desc()).all()

    return render_template("history.html", records=records)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("landing"))



with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True)