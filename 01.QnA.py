from transformers import pipeline

# Load a pre-trained question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Console-based loop for Q&A
def qna_app():
    print("Console Based QnA Application")    
    
    # Provide context for answering questions
    context = input("\nProvide context (a passage of text):\n")
    
    while True:
        question = input("\nAsk a question (or type 'exit' to quit):\n")
        if question.lower() == 'exit':
            print("Exiting QnA application.")
            break
        
        # Get the answer from the model
        result = qa_pipeline(question=question, context=context)
        
        # Print the answer
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['score']:.4f}")

if __name__ == "__main__":
    qna_app()
