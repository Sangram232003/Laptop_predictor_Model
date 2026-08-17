from flask import Flask, request, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Model.pkl"
)

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    MODEL_LOADED = True
    print("Model loaded successfully!")

except Exception as e:
    model = None
    MODEL_LOADED = False
    print("Error loading model:", e)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURE_NAMES = [
    "Age",
    "Gender",
    "Region",
    "Occupation",
    "Income"
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

<title>Laptop Purchase Predictor</title>


<style>

/* =========================================================
   GLOBAL
========================================================= */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family:
        Arial,
        Helvetica,
        sans-serif;
}


body {

    min-height: 100vh;

    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(59,130,246,0.25),
            transparent 30%
        ),

        radial-gradient(
            circle at 90% 90%,
            rgba(139,92,246,0.25),
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #07111f,
            #111827,
            #0f172a
        );

    color: white;

    display: flex;

    justify-content: center;

    align-items: center;

    padding: 30px 15px;

}


/* =========================================================
   MAIN CONTAINER
========================================================= */

.container {

    width: 100%;

    max-width: 1050px;

    background:
        rgba(255,255,255,0.075);

    border:
        1px solid
        rgba(255,255,255,0.12);

    backdrop-filter:
        blur(22px);

    -webkit-backdrop-filter:
        blur(22px);

    border-radius: 28px;

    padding: 42px;

    box-shadow:
        0 30px 80px
        rgba(0,0,0,0.45);

}


/* =========================================================
   HEADER
========================================================= */

.header {

    text-align: center;

    margin-bottom: 35px;

}


.logo {

    width: 75px;

    height: 75px;

    margin:
        0 auto 18px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 21px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );

    font-size: 32px;

    box-shadow:
        0 15px 40px
        rgba(37,99,235,0.40);

}


h1 {

    font-size: 34px;

    font-weight: 800;

    margin-bottom: 12px;

}


.subtitle {

    max-width: 700px;

    margin: auto;

    color: #aeb8cc;

    line-height: 1.6;

    font-size: 15px;

}


/* =========================================================
   MODEL STATUS
========================================================= */

.status {

    display: inline-flex;

    align-items: center;

    gap: 8px;

    margin-top: 18px;

    padding: 8px 17px;

    border-radius: 30px;

    background:
        rgba(34,197,94,0.10);

    border:
        1px solid
        rgba(34,197,94,0.20);

    color: #86efac;

    font-size: 13px;

    font-weight: 600;

}


.status-dot {

    width: 8px;

    height: 8px;

    border-radius: 50%;

    background: #22c55e;

    box-shadow:
        0 0 12px
        rgba(34,197,94,0.8);

}


/* =========================================================
   FORM
========================================================= */

.form-title {

    font-size: 19px;

    font-weight: 700;

    color: #e5e7eb;

    margin-bottom: 22px;

}


.form-grid {

    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 22px;

}


.input-group {

    display: flex;

    flex-direction: column;

}


label {

    color: #d6dbea;

    font-size: 14px;

    font-weight: 600;

    margin-bottom: 9px;

}


.input-help {

    color: #718096;

    font-size: 11px;

    margin-top: 6px;

}


input,
select {

    width: 100%;

    height: 50px;

    padding:
        0 15px;

    border-radius: 13px;

    border:
        1px solid
        rgba(255,255,255,0.12);

    background:
        rgba(255,255,255,0.065);

    color: white;

    outline: none;

    font-size: 14px;

    transition: 0.25s;

}


input::placeholder {

    color: #69758d;

}


select option {

    background: #111827;

    color: white;

}


input:focus,
select:focus {

    border-color: #60a5fa;

    background:
        rgba(255,255,255,0.10);

    box-shadow:
        0 0 0 4px
        rgba(59,130,246,0.12);

}


/* =========================================================
   BUTTON
========================================================= */

.predict-btn {

    width: 100%;

    height: 56px;

    margin-top: 28px;

    border: none;

    border-radius: 15px;

    cursor: pointer;

    color: white;

    font-size: 16px;

    font-weight: 700;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );

    box-shadow:
        0 15px 30px
        rgba(37,99,235,0.30);

    transition: 0.25s;

}


.predict-btn:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 20px 40px
        rgba(37,99,235,0.45);

}


/* =========================================================
   RESULT
========================================================= */

.result {

    margin-top: 30px;

    padding: 32px;

    text-align: center;

    border-radius: 20px;

    background:
        rgba(255,255,255,0.055);

    border:
        1px solid
        rgba(255,255,255,0.10);

}


.result-label {

    color: #9ca8bd;

    font-size: 12px;

    font-weight: 700;

    letter-spacing: 1.5px;

    margin-bottom: 13px;

}


.prediction {

    font-size: 40px;

    font-weight: 800;

}


.prediction.yes {

    color: #4ade80;

    text-shadow:
        0 0 25px
        rgba(74,222,128,0.25);

}


.prediction.no {

    color: #fb7185;

    text-shadow:
        0 0 25px
        rgba(251,113,133,0.20);

}


.result-message {

    margin-top: 10px;

    color: #aab4c7;

    font-size: 14px;

}


/* =========================================================
   ERROR
========================================================= */

.error {

    margin-top: 25px;

    padding: 16px;

    border-radius: 13px;

    background:
        rgba(239,68,68,0.10);

    border:
        1px solid
        rgba(239,68,68,0.20);

    color: #fca5a5;

    text-align: center;

    font-size: 14px;

}


/* =========================================================
   FOOTER
========================================================= */

.footer {

    margin-top: 30px;

    text-align: center;

    color: #667085;

    font-size: 12px;

}


.footer span {

    color: #60a5fa;

}


/* =========================================================
   MOBILE
========================================================= */

@media(max-width:700px) {

    body {

        padding: 15px;

    }

    .container {

        padding:
            28px 20px;

        border-radius: 22px;

    }

    h1 {

        font-size: 27px;

    }

    .form-grid {

        grid-template-columns: 1fr;

        gap: 18px;

    }

    .logo {

        width: 62px;

        height: 62px;

        font-size: 26px;

    }

}

</style>

</head>


<body>


<div class="container">


<!-- =====================================================
     HEADER
===================================================== -->

<div class="header">


    <div class="logo">

        💻

    </div>


    <h1>

        Laptop Purchase Predictor

    </h1>


    <p class="subtitle">

        Predict whether a person is likely to purchase
        a laptop using machine learning based on
        personal, demographic and financial information.

    </p>


    {% if model_loaded %}

    <div class="status">

        <span class="status-dot"></span>

        Model Ready

    </div>

    {% else %}

    <div class="status"
         style="
            color:#fca5a5;
            background:rgba(239,68,68,0.10);
            border-color:rgba(239,68,68,0.20);
         ">

        <span class="status-dot"
              style="background:#ef4444;">
        </span>

        Model Not Loaded

    </div>

    {% endif %}


</div>


<!-- =====================================================
     FORM
===================================================== -->

<div class="form-title">

    Enter Customer Information

</div>


<form method="POST">


<div class="form-grid">


<!-- AGE -->

<div class="input-group">

    <label for="age">

        Age

    </label>

    <input
        type="number"
        id="age"
        name="age"
        min="1"
        max="100"
        placeholder="Enter age"
        required
    >

    <div class="input-help">

        Enter age between 1 and 100

    </div>

</div>


<!-- GENDER -->

<div class="input-group">

    <label for="gender">

        Gender

    </label>

    <select
        id="gender"
        name="gender"
        required
    >

        <option value="">

            Select gender

        </option>

        <option value="0">

            Gender Category 0

        </option>

        <option value="1">

            Gender Category 1

        </option>

    </select>

    <div class="input-help">

        Encoded value used by the ML model

    </div>

</div>


<!-- REGION -->

<div class="input-group">

    <label for="region">

        Region

    </label>

    <select
        id="region"
        name="region"
        required
    >

        <option value="">

            Select region

        </option>

        <option value="0">

            Region Category 0

        </option>

        <option value="1">

            Region Category 1

        </option>

    </select>

    <div class="input-help">

        Encoded value used by the ML model

    </div>

</div>


<!-- OCCUPATION -->

<div class="input-group">

    <label for="occupation">

        Occupation

    </label>

    <select
        id="occupation"
        name="occupation"
        required
    >

        <option value="">

            Select occupation

        </option>

        <option value="0">

            Occupation Category 0

        </option>

        <option value="1">

            Occupation Category 1

        </option>

        <option value="2">

            Occupation Category 2

        </option>

        <option value="3">

            Occupation Category 3

        </option>

    </select>

    <div class="input-help">

        Encoded value used by the ML model

    </div>

</div>


<!-- INCOME -->

<div class="input-group">

    <label for="income">

        Income

    </label>

    <input
        type="number"
        id="income"
        name="income"
        min="0"
        step="any"
        placeholder="Enter income"
        required
    >

    <div class="input-help">

        Enter the income value

    </div>

</div>


</div>


<!-- BUTTON -->

<button
    type="submit"
    class="predict-btn">

    🔮 Predict Laptop Purchase

</button>


</form>


<!-- =====================================================
     RESULT
===================================================== -->

{% if prediction is not none %}


<div class="result">


    <div class="result-label">

        LAPTOP PURCHASE PREDICTION

    </div>


    {% if prediction == "yes" %}


    <div class="prediction yes">

        YES ✓

    </div>


    <div class="result-message">

        This person is likely to purchase a laptop.

    </div>


    {% else %}


    <div class="prediction no">

        NO ✕

    </div>


    <div class="result-message">

        This person is unlikely to purchase a laptop.

    </div>


    {% endif %}


</div>


{% endif %}


<!-- =====================================================
     ERROR
===================================================== -->

{% if error %}


<div class="error">

    ⚠️ {{ error }}

</div>


{% endif %}


<!-- FOOTER -->

<div class="footer">

    Powered by
    <span>Machine Learning</span>
    • Flask • Render

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

        if not MODEL_LOADED:

            error = (
                "Machine learning model could not be loaded."
            )

            return render_template_string(
                HTML,
                model_loaded=False,
                prediction=None,
                error=error
            )


        try:

            # ------------------------------------------------
            # GET VALUES FROM FORM
            # ------------------------------------------------

            age = request.form.get("age")

            gender = request.form.get("gender")

            region = request.form.get("region")

            occupation = request.form.get("occupation")

            income = request.form.get("income")


            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if not age:

                raise ValueError(
                    "Please enter Age."
                )


            if not gender:

                raise ValueError(
                    "Please select Gender."
                )


            if not region:

                raise ValueError(
                    "Please select Region."
                )


            if not occupation:

                raise ValueError(
                    "Please select Occupation."
                )


            if not income:

                raise ValueError(
                    "Please enter Income."
                )


            # ------------------------------------------------
            # CONVERT TO NUMERIC VALUES
            # ------------------------------------------------

            age = float(age)

            gender = float(gender)

            region = float(region)

            occupation = float(occupation)

            income = float(income)


            # ------------------------------------------------
            # VALIDATE AGE
            # ------------------------------------------------

            if age < 1 or age > 100:

                raise ValueError(
                    "Age must be between 1 and 100."
                )


            # ------------------------------------------------
            # VALIDATE INCOME
            # ------------------------------------------------

            if income < 0:

                raise ValueError(
                    "Income cannot be negative."
                )


            # ------------------------------------------------
            # CREATE INPUT ARRAY
            # ------------------------------------------------

            input_data = np.array([
                age,
                gender,
                region,
                occupation,
                income
            ]).reshape(1, -1)


            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            prediction_result = model.predict(
                input_data
            )[0]


            prediction = str(
                prediction_result
            ).lower()


            # ------------------------------------------------
            # CHECK RESULT
            # ------------------------------------------------

            if prediction not in ["yes", "no"]:

                prediction = str(
                    prediction_result
                ).lower()


        except Exception as e:

            error = str(e)


    return render_template_string(

        HTML,

        model_loaded=MODEL_LOADED,

        prediction=prediction,

        error=error

    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")

def health():

    return {

        "status": "healthy",

        "model_loaded": MODEL_LOADED

    }


# ============================================================
# START APPLICATION
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
