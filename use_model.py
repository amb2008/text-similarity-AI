import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Download NLTK data (only required once)
nltk.download('punkt')
nltk.download('stopwords')

# New sentences for prediction (unseen data)
new_sentences = [
  "The pitcher throws to the batter who wants to get a hit in the outfield",
  "If you get a home run, the crowd cheers",
  "I am watching a video about a driving test",
  "I love jumping waves in the ocean in the summer"
]

# Text Preprocessing
def preprocess_text(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

print("Step 1: Preprocessing new sentences...")
# Preprocess the new sentences
preprocessed_new_sentences = [preprocess_text(sentence) for sentence in new_sentences]
print("New sentence preprocessing complete!")

# Load the trained model and vectorizer from the files
model = joblib.load('trained_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Use the loaded vectorizer to transform the new sentences into feature vectors
X_new = vectorizer.transform(preprocessed_new_sentences)

# Use the loaded model to predict labels for the new sentences
predicted_labels = model.predict(X_new)

# Print the predictions
print("Predicted Labels for New Sentences:")
for sentence, label in zip(new_sentences, predicted_labels):
    print(f"Sentence: '{sentence}' --> Predicted Label: {label}")
