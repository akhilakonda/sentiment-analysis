import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the emotion detection model
tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")

# Map the prediction indices to emotion labels with scores
# Negative emotions have negative scores, positive emotions have positive scores
emotion_mapping = {
    0: {"label": "sadness", "score": -2},    # negative emotion
    1: {"label": "joy", "score": 2},         # positive emotion
    2: {"label": "love", "score": 2},        # positive emotion
    3: {"label": "anger", "score": -1},      # negative emotion
    4: {"label": "fear", "score": -2},       # negative emotion
    5: {"label": "surprise", "score": 0}     # neutral emotion
}

def predict_emotion(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get the model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    # Map the prediction to an emotion label and score
    predicted_emotion = emotion_mapping[predicted_class_id]["label"]
    emotion_score = emotion_mapping[predicted_class_id]["score"]

    return predicted_emotion, emotion_score

def evaluate_depression_risk(total_score):
    """Evaluate depression risk based on total emotional score"""
    if total_score < 4:
        return "Low risk of depression. Continue to monitor your emotional well-being."
    elif total_score >= 4 and total_score < 7:
        return "Moderate risk of depression. Consider implementing self-care strategies and talking to someone you trust."
    else:  # score >= 7
        return "Higher risk of depression. It's recommended that you consult with a mental health professional for a proper assessment."

def main():
    print("Depression Screening System")
    print("==========================")
    print("Please answer the following questions about how you've been feeling recently.")
    print("Your responses will be analyzed for emotional content.\n")

    questions = [
        "How have you been feeling emotionally over the past few days?",
        "Do you feel motivated or energetic to do the things you normally enjoy?",
        "Can you describe your sleep routine lately? Are you sleeping too much or too little?",
        "Is there anything that's been on your mind a lot lately or causing you stress?",
        "What's something that made you smile or feel good recently?"
    ]

    emotion_results = []
    total_score = 0

    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        user_input = input("Your response: ")

        if user_input.strip() == "":
            print("No response detected. Please provide an answer.")
            continue

        emotion, score = predict_emotion(user_input)
        emotion_results.append({"question": i+1, "emotion": emotion, "score": score})

        # For this screening tool, we'll invert scores for questions 2 and 5
        # as negative responses to positive questions indicate depression
        if i+1 in [2, 5]:
            total_score -= score
        else:
            total_score += score

    print("\n\nScreening Results:")
    print("=================")
    print("Emotional Analysis:")
    for result in emotion_results:
        print(f"Question {result['question']}: Detected emotion: {result['emotion']} (Score: {result['score']})")

    print(f"\nTotal Depression Risk Score: {total_score}")
    recommendation = evaluate_depression_risk(total_score)
    print(f"\nRecommendation: {recommendation}")

    print("\nPlease note: This is a simplified screening tool and not a clinical diagnosis.")
    print("If you're concerned about your mental health, please consult with a healthcare professional.")

if __name__ == "__main__":
    main()
