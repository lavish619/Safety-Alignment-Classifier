import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
from model import ToxicityClassifier, save_model
from dataset import get_dataloader, get_dataset
import argparse

def train(model, train_loader, optimizer, criterion, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        total_loss = 0
        correct = 0
        total = 0
        for batch in loop:

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=total_loss/total, accuracy=correct/total)

    print(f"Final Training Accuracy: {correct/total:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a toxicity classifier")
    parser.add_argument("--data_directory", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save the model file.")
    args = parser.parse_args()

    data_directory = args.data_directory
    model_path = args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(data_directory, dataset_size=5000, split='train', class_balancing=True)
    dataloader = get_dataloader(dataset)

    model = ToxicityClassifier().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    train(model, dataloader, optimizer, criterion, device, epochs=3)
    save_model(model, model_path)
