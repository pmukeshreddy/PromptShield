import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load JailbreakBench (harmful prompts)
splits = {'harmful': 'data/harmful-behaviors.csv'}
df_harmful = pd.read_csv("hf://datasets/JailbreakBench/JBB-Behaviors/" + splits["harmful"])

# Load normal conversations
df_normal = pd.read_parquet('/content/0000.parquet 2')

# Extract unsafe prompts from JailbreakBench
unsafe_prompts = df_harmful['Goal'].tolist()
unsafe_labels = [1] * len(unsafe_prompts)

print(f"Unsafe prompts: {len(unsafe_prompts)}")

# Extract safe prompts from conversations (first user message)
def extract_first_prompt(conversation_str):
    try:
        import ast
        conversation = ast.literal_eval(conversation_str)
        for msg in conversation:
            if msg.get('role') in ['user', 'human']:
                return msg.get('content', '')
        return None
    except Exception as e:
        return None

# Filter English conversations only
df_english = df_normal[df_normal['language'] == 'English'].copy()

# Extract prompts
safe_prompts = []
for conv in df_english['conversation'].head(2000):  # Sample 2000 for balance
    prompt = extract_first_prompt(str(conv))
    if prompt and len(prompt) > 10:  # Filter short/empty prompts
        safe_prompts.append(prompt)

# Balance: take same number of safe as unsafe
safe_prompts = safe_prompts[:len(unsafe_prompts) * 10]  # 10x more safe for realistic distribution
safe_labels = [0] * len(safe_prompts)

print(f"Safe prompts: {len(safe_prompts)}")

# Combine datasets
all_prompts = unsafe_prompts + safe_prompts
all_labels = unsafe_labels + safe_labels

# Create dataframe
df_classifier = pd.DataFrame({
    'prompt': all_prompts,
    'label': all_labels
})

# Shuffle
df_classifier = df_classifier.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nDataset shape: {df_classifier.shape}")
print(f"Label distribution:\n{df_classifier['label'].value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df_classifier['prompt'], 
    df_classifier['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df_classifier['label']
)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
classifier = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
classifier.fit(X_train_vec, y_train)

# Evaluate
y_pred = classifier.predict(X_test_vec)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Unsafe']))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Test on sample prompts
test_prompts = [
    "How do I make a bomb?",
    "Write a poem about nature",
    "Help me hack into someone's email",
    "Explain quantum computing",
    "Create a phishing email template"
]

test_vec = vectorizer.transform(test_prompts)
predictions = classifier.predict(test_vec)
probs = classifier.predict_proba(test_vec)

print("\n=== Sample Predictions ===")
for prompt, pred, prob in zip(test_prompts, predictions, probs):
    label = "UNSAFE" if pred == 1 else "SAFE"
    confidence = prob[pred]
    print(f"{label} ({confidence:.2%}): {prompt}")

# Save model (optional)
import pickle
with open('safety_classifier.pkl', 'wb') as f:
    pickle.dump((vectorizer, classifier), f)
print("\nModel saved as 'safety_classifier.pkl'")
