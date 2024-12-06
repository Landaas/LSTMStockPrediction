from transformers import pipeline
import torch

# Verify CUDA is available
print("Is CUDA available?", torch.cuda.is_available())

# If True, create pipeline on GPU; if not, it will fall back to CPU
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
