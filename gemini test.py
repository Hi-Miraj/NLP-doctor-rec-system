import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the dataset
data = pd.read_csv('med depart.xlsx')

# Preprocessing
data['Symptom'] = data['Symptom'].str.lower().str.replace(r'[^\w\s]', '')

# Encode labels
label_encoder = LabelEncoder()
data['Doctor_Type'] = label_encoder.fit_transform(data['Doctor_Type'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['Symptom'], data['Doctor_Type'], test_size=0.2, random_state=42)

# BERT Tokenizer and encoding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_and_encode(texts, tokenizer, max_len=128):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')

train_encodings = tokenize_and_encode(X_train.tolist(), tokenizer)
test_encodings = tokenize_and_encode(X_test.tolist(), tokenizer)

class SymptomsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SymptomsDataset(train_encodings, y_train)
test_dataset = SymptomsDataset(test_encodings, y_test)

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Predict new symptoms
def predict_doctor(symptom):
    encoding = tokenize_and_encode([symptom], tokenizer)
    inputs = {key: torch.tensor(val) for key, val in encoding.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    return label_encoder.inverse_transform([prediction.item()])[0]

# Example usage
user_input = "chronic headache and dizziness"
recommended_doctor = predict_doctor(user_input)
print(f"Recommended Doctor: {recommended_doctor}")
