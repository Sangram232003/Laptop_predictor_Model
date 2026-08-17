from flask import Flask, request, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "Model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    print("Model loaded successfully!")

except Exception as e:
    model = None
    print("Error loading model:", e)


# ============================================================
# MODEL INFORMATION
# ============================================================

if model is not None:
    try:
        FEATURE_COUNT = model.n_features_in_
    except:
        try:
            FEATURE_COUNT = model[-1].n_features_in_
        except:
            FEATURE_COUNT = 4
else:
    FEATURE_COUNT = 4


# ============================================================
# CHANGE THESE FEATURE NAMES
# ============================================================
# IMPORTANT:
# Replace these with the exact columns used while training
# your model.

FEATURE_NAMES = [
    "Feature 1",
    "Feature 2",
    "Feature 3",
    "Feature 4"
]

# Automatically create names if model needs more features
if len(FEATURE_NAMES) != FEATURE_COUNT:
    FEATURE_NAMES = [
        f"Feature {i + 1}"
        for i in range(FEATURE_COUNT)
    ]


# ============================================================
# HTML + CSS
# ============================================================

HTML = """
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>AI Prediction System</title>

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Arial, Helvetica, sans-serif;
        }

        body {
            min-height: 100vh;

            background:
                radial-gradient(circle at top left,
                    #243b55,
                    transparent 35%),

                radial-gradient(circle at bottom right,
                    #141e30,
                    transparent 40%),

                #0b1220;

            color: white;

            display: flex;
            justify-content: center;
            align-items: center;

            padding: 30px;
        }

        .container {
            width: 100%;
            max-width: 1000px;

            background: rgba(255,255,255,0.08);

            border: 1px solid rgba(255,255,255,0.15);

            backdrop-filter: blur(20px);

            border-radius: 25px;

            padding: 35px;

            box-shadow:
                0 25px 60px rgba(0,0,0,0.45);
        }

        .header {
            text-align: center;

            margin-bottom: 35px;
        }

        .logo {
            width: 65px;
            height: 65px;

            margin: auto;
            margin-bottom: 15px;

            border-radius: 18px;

            display: flex;
            justify-content: center;
            align-items: center;

            font-size: 30px;

            background: linear-gradient(
                135deg,
                #667eea,
                #764ba2
            );

            box-shadow:
                0 10px 30px rgba(118,75,162,0.5);
        }

        h1 {
            font-size: 32px;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #b8c1d9;
            font-size: 15px;
        }

        .model-status {
            display: inline-block;

            margin-top: 15px;

            padding: 7px 15px;

            border-radius: 30px;

            background: rgba(46, 204, 113, 0.15);

            color: #5cff9d;

            font-size: 13px;
        }

        .form-grid {

            display: grid;

            grid-template-columns:
                repeat(2, 1fr);

            gap: 20px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        label {
            margin-bottom: 8px;

            color: #d9def0;

            font-size: 14px;

            font-weight: 600;
        }

        input {

            width: 100%;

            padding: 14px 16px;

            border-radius: 12px;

            border: 1px solid
                    rgba(255,255,255,0.15);

            background: rgba(255,255,255,0.07);

            color: white;

            outline: none;

            font-size: 15px;

            transition: 0.3s;
        }

        input:focus {

            border-color: #8b7cf6;

            background:
                rgba(255,255,255,0.12);

            box-shadow:
                0 0 0 3px
                rgba(139,124,246,0.15);
        }

        input::placeholder {
            color: #7f8aa8;
        }

        .predict-btn {

            width: 100%;

            margin-top: 28px;

            padding: 16px;

            border: none;

            border-radius: 14px;

            cursor: pointer;

            color: white;

            font-size: 16px;

            font-weight: bold;

            background:
                linear-gradient(
                    135deg,
                    #667eea,
                    #764ba2
                );

            box-shadow:
                0 12px 25px
                rgba(102,126,234,0.3);

            transition: 0.3s;
        }

        .predict-btn:hover {

            transform: translateY(-2px);

            box-shadow:
                0 15px 35px
                rgba(102,126,234,0.5);
        }

        .result {

            margin-top: 30px;

            padding: 25px;

            text-align: center;

            border-radius: 18px;

            background:
                rgba(255,255,255,0.08);

            border:
                1px solid
                rgba(255,255,255,0.15);
        }

        .result-title {

            color: #aab3ca;

            font-size: 14px;

            margin-bottom: 10px;
        }

        .prediction {

            font-size: 30px;

            font-weight: bold;

            color: #ffffff;
        }

        .error {

            margin-top: 20px;

            padding: 15px;

            border-radius: 12px;

            background:
                rgba(255,70,70,0.12);

            color: #ff8f8f;

            text-align: center;
        }

        .footer {

            text-align: center;

            margin-top: 25px;

            color: #727c96;

            font-size: 12px;
        }

        @media(max-width: 700px) {

            body {
                padding: 15px;
            }

            .container {
                padding: 25px 18px;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }

            h1 {
                font-size: 26px;
            }
        }

    </style>

</head>


<body>

<div class="container">

    <div class="header">

        <div class="logo">
            AI
        </div>

        <h1>
            AI Prediction System
        </h1>

        <p class="subtitle">
            Enter the required values and let the
            machine learning model generate a prediction.
        </p>

        {% if model_loaded %}

        <div class="model-status">
            ● Model Ready
        </div>

        {% else %}

        <div class="model-status"
             style="color:#ff8f8f;">
            ● Model Not Loaded
        </div>

        {% endif %}

    </div>


    <form method="POST">

        <div class="form-grid">

            {% for feature in features %}

            <div class="input-group">

                <label>
                    {{ feature }}
                </label>

                <input
                    type="number"
                    step="any"
                    name="feature{{ loop.index0 }}"
                    placeholder="Enter {{ feature }}"
                    required
                >

            </div>

            {% endfor %}

        </div>


        <button
            type="submit"
            class="predict-btn">

            🚀 Generate Prediction

        </button>

    </form>


    {% if prediction is not none %}

    <div class="result">

        <div class="result-title">
            MODEL PREDICTION
        </div>

        <div class="prediction">
            {{ prediction }}
        </div>

    </div>

    {% endif %}


    {% if error %}

    <div class="error">
        ⚠ {{ error }}
    </div>

    {% endif %}


    <div class="footer">

        Powered by Machine Learning • Flask • Render

    </div>

</div>

</body>

</html>
"""


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:

            # Collect input values
            values = []

            for i in range(FEATURE_COUNT):

                value = request.form.get(
                    f"feature{i}"
                )

                if value is None or value == "":
                    raise ValueError(
                        f"Missing value for Feature {i + 1}"
                    )

                values.append(float(value))


            # Convert to numpy array
            input_data = np.array(values).reshape(1, -1)


            # Make prediction
            prediction_result = model.predict(
                input_data
            )[0]


            # Format prediction
            if isinstance(
                prediction_result,
                (np.integer, np.floating)
            ):

                prediction = str(
                    prediction_result
                )

            else:

                prediction = str(
                    prediction_result
                )


        except Exception as e:

            error = str(e)


    return render_template_string(

        HTML,

        features=FEATURE_NAMES,

        prediction=prediction,

        error=error,

        model_loaded=model is not None
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
