# Health-Aid-Application
# Medical Image Analysis (Using CNN)
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the dataset (e.g., Kaggle medical image dataset)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'path_to_train_images',
    image_size=(256, 256),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'path_to_val_images',
    image_size=(256, 256),
    batch_size=32
)

# Create CNN Model
model = models.Sequential([
    layers.InputLayer(input_shape=(256, 256, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (for e.g., disease/no disease)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_ds, validation_data=val_ds, epochs=10)

  2. Symptom-Based Diagnosis (NLP Model using BERT)  
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import torch

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Example symptom input
symptoms = "I have a persistent cough and fever."

# Tokenize the input
inputs = tokenizer(symptoms, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Forward pass
outputs = model(**inputs)

# Interpret result
predictions = torch.argmax(outputs.logits, dim=1)
# Medical Record Analysis (Structured Data)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load medical records dataset
data = pd.read_csv('path_to_medical_records.csv')

# Prepare dataset
X = data[['age', 'bmi', 'blood_pressure', 'cholesterol']]  # Feature columns
y = data['disease_label']  # Target column

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train RandomForest model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

  4. Chatbot Interface (Using Rasa or Flask)  
# Install rasa using pip: pip install rasa
import openai

def get_diagnosis(user_input):
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=f"Patient symptoms: {user_input}. Provide diagnosis and treatment recommendations.",
      max_tokens=150
    )
    return response['choices'][0]['text']

# Flask example for chatbot
from flask import Flask, request, jsonify

app = Flask(_name_)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    diagnosis = get_diagnosis(user_input)
    return jsonify({"response": diagnosis})

if _name_ == '_main_':
    app.run(debug=True)
	
	# Integration and System Flow
def diagnose(patient_symptoms, medical_image_path, medical_record):

    # 1. Analyze medical image
    image_diagnosis = analyze_image(medical_image_path)  # Use CNN model for this
   
    # 2. Analyze symptoms
    symptom_diagnosis = analyze_symptoms(patient_symptoms)  # Use BERT or GPT for this
   
    # 3. Analyze medical records
    record_diagnosis = analyze_records(medical_record)  # Use Random Forest/ML model
   
    # Combine results
    final_diagnosis = combine_diagnoses(image_diagnosis, symptom_diagnosis, record_diagnosis)
   
    return final_diagnosis
