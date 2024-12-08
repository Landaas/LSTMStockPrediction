from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

def sentiment(text):
    classification = classifier(text)
    if len(classification):
        result = classification[0]
        score = result['score']
        if result['label'] == 'NEGATIVE':
            score = 1 - score

        return score
