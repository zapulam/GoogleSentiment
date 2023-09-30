from transformers import pipeline
from GoogleNews import GoogleNews
from flask import Flask, render_template, request

# Create a Flask web app instance
app = Flask(__name__)

# Define a route for the root URL ("/")
@app.route("/", methods=["GET", "POST"])
def index():
    user_input = ""
    
    if request.method == "POST":
        user_input = request.form["user_input"]
    
    return render_template("index.html", user_input=user_input)

# Run the app if this script is executed
if __name__ == "__main__":
    app.run(debug=True)