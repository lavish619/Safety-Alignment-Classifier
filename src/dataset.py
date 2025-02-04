
import torch
from torch.utils.data import Dataset as nn_Dataset
from torch.utils.data import DataLoader
import random
import string
from transformers import MarianMTModel, MarianTokenizer
import spacy
from datasets import load_dataset, Dataset
from transformers import RobertaTokenizer
import pandas as pd
import random
nlp = spacy.load("en_core_web_md")

# Convert dataset to PyTorch tensors
class ToxicityDataset(nn_Dataset):
    def __init__(self, dataset, transform = None):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Apply transformation (e.g., adversarial attack) if provided
        text = self.dataset[idx]['comment_text']
        label = torch.tensor(self.dataset[idx]["label"], dtype=torch.long)
        is_adversarial = torch.tensor(0, dtype = torch.bool)
        
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")

        # Convert tokenized outputs to tensor format
        item = {
            'text': text, 
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label, 
            'is_adversarial' : is_adversarial
        }
        return item

KEYBOARD_MAP = {
    'q': 'qw', 'w': 'weq', 'e': 'wer', 'r': 'ret', 't': 'tyr', 'y': 'ytu', 'u': 'uio', 'i': 'iop', 'o': 'op',
    'p': 'plo', 'a': 'as', 's': 'sad', 'd': 'dsf', 'f': 'fdg', 'g': 'gfh', 'h': 'hgj', 'j': 'jhk', 'k': 'kjl',
    'l': 'lk', 'z': 'zx', 'x': 'xzc', 'c': 'cvx', 'v': 'vbc', 'b': 'bnv', 'n': 'nbm', 'm': 'mn'
}

def keyboard_typo(text, p=0.1):
    words = text.split()
    perturbed_words = []
    for word in words:
        if random.random() < p and len(word) > 1:
            i = random.randint(0, len(word) - 1)
            if word[i].lower() in KEYBOARD_MAP:
                new_char = random.choice(KEYBOARD_MAP[word[i].lower()])
                word = word[:i] + new_char + word[i+1:]
        perturbed_words.append(word)
    return ' '.join(perturbed_words)

def random_char_insertion(text, p=0.1):
    words = text.split()
    perturbed_words = []
    for word in words:
        if random.random() < p:
            i = random.randint(0, len(word))
            char = random.choice(string.ascii_lowercase)
            word = word[:i] + char + word[i:]
        perturbed_words.append(word)
    return ' '.join(perturbed_words)

# def synonym_replacement(text, p=0.2):
#     words = text.split()
#     perturbed_words = []
#     for word in words:
#         if random.random() < p:
#             synonyms = wordnet.synsets(word)
#             if synonyms:
#                 lemmas = synonyms[0].lemmas()
#                 if len(lemmas) > 1:
#                     word = random.choice(lemmas[1:]).name().replace('_', ' ')
#         perturbed_words.append(word)
#     return ' '.join(perturbed_words)
    
def synonym_replacement(text, p=0.2, top_k = 5):
    words = text.split()
    perturbed_words = []
    
    for word in words:
        if random.random() < p:
            token = nlp(word)[0]
            if token.has_vector: 
                similar_words = sorted([(w.text, token.similarity(w)) for w in nlp.vocab if w.has_vector and w.is_alpha and not w.is_stop and w.text != token.text and w.vector_norm > 0], 
                        key=lambda item: item[1], 
                        reverse=True
                    )
                if similar_words:
                    word = random.choice(similar_words[:top_k])[0]
        perturbed_words.append(word)
    return ' '.join(perturbed_words)

HOMOPHONE_MAP = {
    'to': 'too', 'too': 'two', 'there': 'their', 'their': 'they\'re', 'hear': 'here',
    'where': 'wear', 'know': 'no', 'new': 'knew', 'right': 'write', 'bare': 'bear'
}

def homophone_substitution(text, p=0.2):
    words = text.split()
    perturbed_words = [HOMOPHONE_MAP.get(word, word) if random.random() < p else word for word in words]
    return ' '.join(perturbed_words)

def word_order_shuffle(text, window_size=2):
    words = text.split()
    n = len(words)
    if n < 2:
        return text  
    shuffled_words = words.copy()
    for i in range(n - 1):
        if random.random() < 0.5: 
            swap_idx = min(i + random.randint(1, window_size), n - 1)
            shuffled_words[i], shuffled_words[swap_idx] = shuffled_words[swap_idx], shuffled_words[i]
    return ' '.join(shuffled_words)

def translate_text(text, src_lang="en", tgt_lang="fr"):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def back_translation(text):
    try:
        fr_text = translate_text(text, "en", "fr")
        return translate_text(fr_text, "fr", "en")
    except Exception:
        return text 

class AdversarialTextDataset(nn_Dataset):
    def __init__(self, original_dataset, transform_prob=0.3, apply_back_translation=False):
        """
        :param original_dataset: List of tuples (text, label) or any dataset with (text, label) format.
        :param transform_prob: Probability of applying each transformation.
        :param apply_back_translation: Whether to use back translation (slow).
        """
        self.original_dataset = original_dataset
        self.transform_prob = transform_prob
        self.apply_back_translation = apply_back_translation
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def adversarial_transform(self, text):
        """Applies multiple perturbations randomly."""
        if random.random() < self.transform_prob:
            text = keyboard_typo(text, self.transform_prob)
        if random.random() < self.transform_prob:
            text = random_char_insertion(text, self.transform_prob)
        if random.random() < self.transform_prob:
            text = synonym_replacement(text, self.transform_prob)
        if random.random() < self.transform_prob:
            text = homophone_substitution(text, self.transform_prob)
        if random.random() < self.transform_prob:
            text = word_order_shuffle(text, window_size=2)
        if self.apply_back_translation and random.random() < self.transform_prob:
            text = back_translation(text)
        return text

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        text = self.original_dataset[idx]['text']
        label = self.original_dataset[idx]['label']
        perturbed_text = self.adversarial_transform(text)
        is_adversarial = torch.tensor(1, dtype = torch.bool)

        # Tokenize on the fly
        encoding = self.tokenizer(perturbed_text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")

        # Convert tokenized outputs to tensor format
        item = {
            'text': perturbed_text, 
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label, 
            'is_adversarial': is_adversarial
        }
        return item

def preprocess_function(example):
    example["label"] = 1 if example["toxic"] >= 0.5 else 0
    return example

def load_toxicity_dataset(train_size, test_size, class_balancing):

    dataset = load_dataset("jigsaw_toxicity_pred", data_dir="./jigsaw_toxicity_data", trust_remote_code=True)
    dataset = dataset.map(preprocess_function)

    if class_balancing:
        ## train set
        df_train = pd.DataFrame(dataset["train"])
        df_label_0 = df_train[df_train["label"] == 0]
        df_label_1 = df_train[df_train["label"] == 1]
        df_balanced = pd.concat([
            df_label_0.sample(n=train_size//2, random_state=42),
            df_label_1.sample(n=train_size//2, random_state=42)
        ])
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        dataset["train"] = Dataset.from_pandas(df_balanced)

        ## test set
        df_test = pd.DataFrame(dataset["test"])
        df_label_0_test, df_label_1_test = df_test[df_test["label"] == 0], df_train[df_train["label"] == 1]
        df_balanced_test = pd.concat([
            df_label_0_test.sample(n=test_size//2, random_state=42),
            df_label_1_test.sample(n=test_size//2, random_state=42)
        ])
        df_balanced_test = df_balanced_test.sample(frac=1, random_state=42).reset_index(drop=True)
        dataset["test"] = Dataset.from_pandas(df_balanced_test)
        print(dataset)
    else:
        # Shuffle dataset and sample
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(train_size))
        dataset["test"] = dataset["test"].shuffle(seed=42).select(range(test_size))

    train_dataset = ToxicityDataset(dataset["train"])
    test_dataset = ToxicityDataset(dataset["test"])
    print(len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset

def get_dataloaders(train_dataset, 
                    test_dataset,
                    batch_size: int  = 32,
                    shuffle: bool = True, 
                    drop_last: bool = False):
    
    if train_dataset:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    else:
        train_loader = None
    
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    else:
        test_loader = None
    return train_loader, test_loader

def get_adversarial_dataloader(original_dataset,
                                batch_size: int  = 32,
                                shuffle: bool = True, 
                                drop_last: bool = False,
                                transform_prob: float= 0.4,
                                apply_back_translation: bool = False
                                ):
    # Wrap the dataset with adversarial transformations
    adv_dataset = AdversarialTextDataset(original_dataset, transform_prob=transform_prob, apply_back_translation=apply_back_translation)
    adv_dataloader = DataLoader(adv_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return adv_dataset, adv_dataloader