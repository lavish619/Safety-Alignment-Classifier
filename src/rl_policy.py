import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from dataset import get_adversarial_dataloader, get_dataset
import argparse
from model import ToxicityClassifier, save_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x) 
        return logits

class PPOAgent:
    def __init__(self, classifier, policy_network, epsilon=0.2):
        self.classifier = classifier
        self.policy = policy_network
        self.epsilon = epsilon 

    def select_action(self, state):
        logits = self.policy(state)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1)
        return action, probs.gather(1, action)

    def compute_loss(self, old_probs, new_probs, rewards):
        rewards = rewards.unsqueeze(1)
        ratio = new_probs / (old_probs + 1e-8)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(ratio * rewards, clipped_ratio * rewards).mean()
        return loss        

    def compute_reward(self, input_ids, attention_mask, labels, adversarial):
        with torch.no_grad():
            logits = self.classifier(input_ids, attention_mask) 
            predictions = logits.argmax(dim=-1) 

        correct = (predictions == labels).float()
        reward = torch.where(adversarial, -correct + 1, correct)
        return reward  

# Environment simulation
def train_rl_classifier(model, policy_network, adv_dataloader, epochs=4):
    policy_network.train()
    agent = PPOAgent(model, policy_network)
    optimizer = optim.Adam(policy_network.parameters(), lr=3e-4)
    
    for epoch in range(epochs):
        loop = tqdm(adv_dataloader, leave=True)
        total_loss = 0
        correct = 0
        total = 0
        total_reward = 0
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            is_adversarial = batch['is_adversarial'].to(device)

            optimizer.zero_grad()
            # Using model logits as states of environment
            states = model(input_ids, attention_mask).detach()  
            actions, old_probs = agent.select_action(states)

            rewards = agent.compute_reward(model, input_ids, attention_mask, labels, is_adversarial)
            new_logits = agent.policy(states)
            new_probs = F.softmax(new_logits, dim=-1).gather(1, actions)
            loss = agent.compute_loss(old_probs, new_probs, rewards)
    
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_reward += rewards.sum().item()
            preds = torch.argmax(states, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(policyloss=total_loss/total, accuracy=correct/total, rewards = total_reward/total)
            
    return agent

def evaluate_rl_classifier(classifier, policy_network, dataloader, device):
    classifier.eval()  
    policy_network.eval() 
    
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
            )

            logits = classifier(input_ids, attention_mask)
            policy_adjustments = policy_network(logits)
            adjusted_logits = logits + policy_adjustments
            predictions = adjusted_logits.argmax(dim=-1)

            preds_list.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    accuracy = accuracy_score(labels_list, preds_list)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_list, preds_list, average="binary")

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RL based policy for a pretrained toxicity classifier")
    parser.add_argument("--data_directory", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--classifier_model_path", type=str, required=True, help="Path to load the classifier model file.")
    parser.add_argument("--policy_model_path", type=str, required=True, help="Path to save the policy model file.")
    args = parser.parse_args()

    data_directory = args.data_directory
    classifier_model_path = args.classifier_model_path
    policy_model_path = args.policy_model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ToxicityClassifier()
    model.load_state_dict(torch.load(classifier_model_path, weights_only=True, map_location=torch.device('cpu')))
    
    policy_network = PolicyNetwork(input_dim=2, num_classes=2)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        policy_network = torch.nn.DataParallel(policy_network)
    model = model.to(device)
    policy_network = policy_network.to(device)

    ## train
    train_dataset = get_dataset(data_directory, dataset_size=5000, split='train', class_balancing=True)
    adv_train_dataloader = get_adversarial_dataloader(train_dataset, batch_size=64, transform_prob=0.4)
    agent = train_rl_classifier(model, policy_network, adv_train_dataloader)
    save_model(agent.policy, policy_model_path)
    
    ## evaluate
    test_dataset = get_dataset(data_directory, dataset_size=1000, split='test', class_balancing=True)
    adv_test_dataloader = get_adversarial_dataloader(test_dataset, batch_size=64, transform_prob=0.4)
    evaluate_rl_classifier(model, agent.policy, adv_test_dataloader, device)






