import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from langchain_ollama import OllamaLLM

# Load user interactions
def load_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Preprocess data for LSTM
def preprocess_data(user_data):
    sequences = []
    targets = []
    incorrect_topics = []
    for entry in user_data:
        time_spent = entry["time_spent"]
        correct = 1 if entry["response"]["correct_answer"] in entry["correct_answers"] else 0
        sequences.append([time_spent, correct])
        targets.append(correct)  # Predict memory retention
        if not correct:
            incorrect_topics.append(entry["query"])
    return np.array(sequences), np.array(targets), incorrect_topics

# Define LSTM model
class MemoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MemoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# Training the LSTM model
def train_lstm(model, train_loader, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Initialize LLM
llm = OllamaLLM(model="olmo2", base_url="http://localhost:11434", temperature=0.1)

def generate_recommendations(incorrect_topics):
    if not incorrect_topics:
        return "Great job! No weak areas detected."
    return f"Focus on these topics: {', '.join(set(incorrect_topics))}"

# Main execution
if __name__ == "__main__":
    user_data = load_data("user_interactions.json")
    sequences, targets, incorrect_topics = preprocess_data(user_data)
    sequences = torch.tensor(sequences, dtype=torch.float32).unsqueeze(1)
    targets = torch.tensor(targets, dtype=torch.float32)
    dataset = TensorDataset(sequences, targets)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Train LSTM
    model = MemoryLSTM(input_size=2, hidden_size=8, output_size=1)
    train_lstm(model, train_loader)
    
    # Predict memory loss
    with torch.no_grad():
        predictions = model(sequences).squeeze().numpy()
    memory_scores = (100 * predictions).tolist()
    
    # Generate topic-based recommendations
    recommendations = generate_recommendations(incorrect_topics)
    print("Recommended focus areas:", recommendations)
