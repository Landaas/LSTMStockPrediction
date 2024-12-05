from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def sentiment(text):
    classification = classifier(text)
    if len(classification):
        result = classification[0]
        score = result['score']
        if result['label'] == 'NEGATIVE':
            score = 1 - score

        return score