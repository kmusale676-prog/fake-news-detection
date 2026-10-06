from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# -----------------------------------
# TRAINING DATA
# -----------------------------------

news_text = [
    "NASA finds water on Mars",
    "Apple launches new iPhone with AI",
    "Airlines have landed in Delhi",
    "Government approves new education policy",
    "COVID-19 vaccine approved worldwide",

    "World ends tomorrow",
    "Elon Musk buys the moon",
    "Aliens landed in Mumbai yesterday",
    "Scientists discover a magical invisible city",
    "Moon crashes into Earth tomorrow"
]

labels = [
    "Real",
    "Real",
    "Real",
    "Real",
    "Real",

    "Fake",
    "Fake",
    "Fake",
    "Fake",
    "Fake"
]

# -----------------------------------
# MACHINE LEARNING MODEL
# -----------------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(news_text)

model = LogisticRegression()

model.fit(X, labels)


# -----------------------------------
# WEBSITE HTML
# -----------------------------------

HTML = """
<!DOCTYPE html>

<html>

<head>

    <title>Fake News Detection</title>

    <style>

        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea, #764ba2);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            width: 90%;
            max-width: 700px;
            background: white;
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        h1 {
            color: #333;
            font-size: 35px;
            margin-bottom: 10px;
        }

        p {
            color: #666;
        }

        textarea {
            width: 95%;
            height: 150px;
            padding: 15px;
            margin-top: 20px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
            resize: none;
        }

        button {
            margin-top: 20px;
            padding: 15px 35px;
            border: none;
            border-radius: 10px;
            background: #667eea;
            color: white;
            font-size: 17px;
            cursor: pointer;
        }

        button:hover {
            background: #764ba2;
        }

        .real {
            margin-top: 25px;
            padding: 20px;
            border-radius: 10px;
            background: #d4edda;
            color: #155724;
            font-size: 22px;
            font-weight: bold;
        }

        .fake {
            margin-top: 25px;
            padding: 20px;
            border-radius: 10px;
            background: #f8d7da;
            color: #721c24;
            font-size: 22px;
            font-weight: bold;
        }

    </style>

</head>


<body>

    <div class="container">

        <h1>📰 Fake News Detection</h1>

        <p>
            Enter a news headline below to check the prediction.
        </p>

        <form method="POST">

            <textarea
                name="news"
                placeholder="Enter news headline here..."
                required>{{ news }}</textarea>

            <br>

            <button type="submit">
                🔍 Check News
            </button>

        </form>


        {% if prediction == "Fake" %}

            <div class="fake">
                ❌ This News is FAKE
            </div>

        {% elif prediction == "Real" %}

            <div class="real">
                ✅ This News is REAL
            </div>

        {% endif %}

    </div>

</body>

</html>
"""


# -----------------------------------
# HOME PAGE
# -----------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    news = ""

    if request.method == "POST":

        news = request.form["news"]

        # Convert text into numbers
        news_vector = vectorizer.transform([news])

        # Predict
        prediction = model.predict(news_vector)[0]

    return render_template_string(
        HTML,
        prediction=prediction,

    
        news=news
    )


# -----------------------------------
# START WEBSITE
# -----------------------------------

if __name__ == "__main__":

    app.run(debug=True)