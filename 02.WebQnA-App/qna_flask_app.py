from flask import Flask, request, render_template
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Home route to render the form
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle QnA requests
@app.route("/answer", methods=["POST"])
def answer():
    context = request.form["context"]
    question = request.form["question"]
    
    # Use the pipeline to get the answer
    result = qa_pipeline(question=question, context=context)
    
    # Extract answer and confidence score
    answer = result['answer']
    confidence = f"{result['score']:.4f}"

    # Render the result page with the answer
    return render_template("result.html", context=context, question=question, answer=answer, confidence=confidence)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)
