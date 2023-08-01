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

# Sample labeled dataset (you should replace these with your actual data)
sentences = [
    "Baseball is America's favorite pastime.",
    "The pitcher threw a fastball at 95 mph.",
    "The baseball game went into extra innings.",
    "He hit a home run over the center field fence.",
    "The crowd cheered as the team won the championship.",
    "The baseball team practiced their swings.",
    "In baseball, three strikes mean you're out.",
    "Baseball players wear gloves to catch the ball.",
    "The baseball diamond has four bases.",
    "The baseball stadium was packed with fans.",
    "Soccer and basketball are popular sports worldwide.",
    "The company's stock price soared to a new record.",
    "The musician played a beautiful melody on the piano.",
    "The book's plot had an unexpected twist.",
    "She captured stunning photos of the sunset.",
    "I need to buy some groceries for the week.",
    "The weather forecast predicts rain tomorrow.",
    "The city hosted a lively music festival.",
    "I enjoy hiking in the mountains on weekends.",
    "The new movie received mixed reviews from critics."
]

#label in correspondence with sentences. 1 if the sentence is what you'd like to recieve as being similar, and 0 if not.
labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Text Preprocessing
def preprocess_text(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

print("Step 1: Preprocessing sentences...")
# Preprocess all sentences in the dataset
preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
print("Sentence preprocessing complete!")

# Feature Extraction - Bag-of-Words
vectorizer = CountVectorizer()

print("Step 2: Extracting features using Bag-of-Words...")
# Transform the preprocessed sentences into feature vectors
X = vectorizer.fit_transform(preprocessed_sentences)
y = labels
print("Feature extraction complete!")

# Model Training - Logistic Regression
print("Step 3: Training the Logistic Regression model...")
model = LogisticRegression()
model.fit(X, y)
print("Model training complete!")

# Save the trained model and vectorizer to files using joblib
joblib.dump(model, 'trained_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
print("Trained model and vectorizer saved.")

