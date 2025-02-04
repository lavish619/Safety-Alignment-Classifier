from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from tqdm import tqdm
from model import ToxicityClassifier
from dataset import get_dataloader, get_dataset, get_adversarial_dataloader
import argparse

def evaluate(model, test_loader, device):
    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    accuracy = accuracy_score(labels_list, preds_list)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_list, preds_list, average="binary")

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a toxicity classifier with optional adversarial examples.")
    parser.add_argument("--data_directory", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--adversarial", action="store_true", help="Enable adversarial evaluation.")
    args = parser.parse_args()

    data_directory = args.data_directory
    model_path = args.model_path
    adversarial = args.adversarial

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(data_directory, dataset_size=1000, split='test', class_balancing=True)
    dataloader = get_dataloader(dataset)

    model = ToxicityClassifier()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model = model.to(device)
    
    evaluate(model, dataloader, device)

    if adversarial:
        _, adv_dataloader = get_adversarial_dataloader(dataset, transform_prob=0.4)
        evaluate(model, adv_dataloader, device)

