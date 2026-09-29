import os
import numpy as np
import pandas as pd
import random
import torch
import argparse
from pathlib import Path
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import login
from huggingface_hub import whoami
from dotenv import load_dotenv
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)
res=whoami()
print(f"\nYou are : {res['name']}")

class MemeFeatureDataset(Dataset):
    def __init__(self, features, l1_labels, l2_labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.l1 = torch.tensor(l1_labels, dtype=torch.long)
        self.l2 = torch.tensor(l2_labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.l1[idx], self.l2[idx]

class HierarchicalClassifier(nn.Module):
    def __init__(self, input_dim, num_l1, num_l2, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
        )
        self.l1_head = nn.Sequential(
            nn.Linear(32, num_l1),
        )
        # Level 2 head sees shared features + level-1 logits (conditioning)
        self.l2_head = nn.Sequential(
            nn.Linear(32 + num_l1, 16),
            nn.ReLU(),
            nn.Linear(16, num_l2),
        )

    def forward(self, x):
        shared = self.shared(x)
        l1_logits = self.l1_head(shared)
        l2_input = torch.cat([shared, l1_logits.detach()], dim=1)  # detach so L2 doesn't backprop noise into L1 head
        l2_logits = self.l2_head(l2_input)
        return l1_logits, l2_logits

def count_parameters(module):
    return sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )

def model_description(model):
    print(
        f"{'Component':<25}"
        f"{'Type':<20}"
        f"{'Configuration':<45}"
        f"{'Parameters':>15}"
    )
    print("-" * 105)
    for name, module in model.named_children():
        params = count_parameters(module)
        print(
            f"{name:<25}"
            f"{module.__class__.__name__:<20}"
            f"{str(module):<45}"
            f"{params:>15,}"
        )
    total = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    print("-" * 105)
    print(f"{'Total trainable':<90}{total:>15,}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="./data/train", help="Path to the training data")

    return parser.parse_args()

def set_seed(SEED):
    # setting seed for reproducibility of the results
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_embeddings(embeddings, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)
    print(f"Saved embeddings: {path}")
    print(f"Shape: {embeddings.shape}")


def load_embeddings(path):
    embeddings = np.load(path)
    print(f"Loaded embeddings: {path}")
    print(f"Shape: {embeddings.shape}")
    return embeddings

def distribution_plot(df, train_df, val_df, seed_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Level1 ---
    total_counts1 = df['Level1'].value_counts()
    train_counts1 = train_df['Level1'].value_counts().reindex(total_counts1.index, fill_value=0)
    val_counts1 = val_df['Level1'].value_counts().reindex(total_counts1.index, fill_value=0)

    labels1 = total_counts1.index
    x = np.arange(len(labels1))
    max_y1 = total_counts1.max()

    bars_total = axes[0].bar(x - 0.25, total_counts1, width=0.25, label='Total')
    bars_train = axes[0].bar(x, train_counts1, width=0.25, label='Train')
    bars_val = axes[0].bar(x + 0.25, val_counts1, width=0.25, label='Validation')
    axes[0].set_ylim(0, max_y1 * 1.08)

    axes[0].set_xticks(x)
    axes[0].set_ylabel("Instances", fontsize=12)
    axes[0].set_xticklabels(labels1, rotation=30, fontweight='bold', fontsize=12)
    axes[0].set_title("Level 1", fontsize=20)
    axes[0].legend()

    # Add labels on bars
    axes[0].bar_label(bars_total, padding=2)
    axes[0].bar_label(bars_train, padding=2)
    axes[0].bar_label(bars_val, padding=2)

    # --- Level2 comparison ---
    total_counts2 = df['Level2'].value_counts()
    train_counts2 = train_df['Level2'].value_counts().reindex(total_counts2.index, fill_value=0)
    val_counts2 = val_df['Level2'].value_counts().reindex(total_counts2.index, fill_value=0)

    labels2 = total_counts2.index
    x = np.arange(len(labels2))
    max_y2 = total_counts2.max()

    bars_total = axes[1].bar(x - 0.25, total_counts2, width=0.25, label='Total')
    bars_train = axes[1].bar(x, train_counts2, width=0.25, label='Train')
    bars_val = axes[1].bar(x + 0.25, val_counts2, width=0.25, label='Validation')
    axes[1].set_ylim(0, max_y2 * 1.08)

    axes[1].set_xticks(x)
    axes[1].set_ylabel("Instances", fontsize=12)
    axes[1].set_xticklabels(labels2, rotation=30, fontweight='bold', fontsize=12)
    axes[1].set_title("Level 2", fontsize=20)
    axes[1].legend()

    # Add labels on bars
    axes[1].bar_label(bars_total, padding=2)
    axes[1].bar_label(bars_train, padding=2)
    axes[1].bar_label(bars_val, padding=2)

    fig.suptitle(f"Train and validation splitting", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(seed_dir, f"label_distribution.png"), dpi=300, bbox_inches='tight')
    # plt.show()

def get_image_embeddings(img_model, paths, IMAGE_DIR, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(paths), batch_size)):
        batch_paths = paths[i:i+batch_size]
        imgs = [Image.open(os.path.join(IMAGE_DIR, p)).convert("RGB") for p in batch_paths]
        emb = img_model.encode(imgs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    return np.vstack(embeddings)

def get_text_embeddings(text_model, texts, batch_size=32):
    return text_model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

# ============================================================
# TEXT EMBEDDING FUNCTIONS
# ============================================================

def get_bge_embeddings(text_model, texts, device, batch_size=32):
    embeddings = []
    text_model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="BGE-M3 embeddings"):
        batch_texts = list(texts[i:i + batch_size])
        with torch.no_grad():
            batch_embeddings = text_model.encode(
                batch_texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False
            )
        # BGE-M3 = 1024 dimensions
        embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)


def get_muril_embeddings(text_model, tokenizer, texts, device, batch_size=32):
    embeddings = []
    text_model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="MuRIL embeddings"):
        batch_texts = list(texts[i:i + batch_size])
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = text_model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"]
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            batch_embeddings = summed / counts
        # MuRIL = 768 dimensions
        embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_DIR = os.path.join(args.train_dir, "train_images")
    TRAIN_EXCEL_FILE = os.path.join(args.train_dir, "train_labels.xlsx")
    df = pd.read_excel(TRAIN_EXCEL_FILE)

    stem_to_file = {}
    for f in os.listdir(IMAGE_DIR):
        stem, _ = os.path.splitext(f)
        stem_to_file[stem] = f

    stems = df['Image_id'].astype(str).str.zfill(3)
    df['image_path'] = stems.map(stem_to_file)

    # Sample of the dataframe
    print(f"Sample of the dataframe: \n{df.sample(5)}")

    # Check for null images
    print(f"\nCount of null images are : {df['image_path'].isnull().sum()}")

    # Label counts for each level
    print(f"\nLevel1 label counts: \n{df['Level1'].value_counts()}")
    print(f"\nLevel2 label counts: \n{df['Level2'].value_counts()}")

    BASE_DIR = Path(__file__).parent
    RESULTS_DIR = BASE_DIR / "results"
    EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, "embeddings")
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # ============================================================
    # IMAGE ENCODER
    # ============================================================

    img_model = SentenceTransformer("clip-ViT-B-32", device=str(device), cache_folder="./hf_models")

    image_embedding_path = EMBEDDINGS_DIR / "image_embeddings.npy"
    if image_embedding_path.exists():
        print("\nLoading precomputed image embeddings...")
        all_img_emb = load_embeddings(image_embedding_path)
    else:
        print("\nGenerating image embeddings...")
        all_img_emb = get_image_embeddings(img_model, df['image_path'].tolist(), IMAGE_DIR, batch_size=32)
        save_embeddings(all_img_emb,image_embedding_path)

    # ============================================================
    # TEXT EMBEDDINGS PER ENCODER
    # ============================================================

    text_embeddings = {}

    for encoder_name in ["bge-m3", "muril"]:
        if encoder_name == "bge-m3":
            text_embedding_path = EMBEDDINGS_DIR / "bge-m3_text_embeddings.npy"
            if text_embedding_path.exists():
                print("\nLoading precomputed BGE-M3 embeddings...")
                all_txt_emb = load_embeddings(text_embedding_path)
            else:
                print("\nGenerating BGE-M3 embeddings...")
                bge_model = SentenceTransformer("BAAI/bge-m3", device=device, cache_folder="./hf_models")
                all_txt_emb = get_bge_embeddings(bge_model, df['Text'].astype(str).tolist(), device, batch_size=32)
                save_embeddings(all_txt_emb, text_embedding_path)
            text_embeddings["bge-m3"] = all_txt_emb
        elif encoder_name == "muril":
            text_embedding_path = EMBEDDINGS_DIR / "muril_text_embeddings.npy"
            if text_embedding_path.exists():
                print("\nLoading precomputed MuRIL embeddings...")
                all_txt_emb = load_embeddings(text_embedding_path)
            else:
                print("\nGenerating MuRIL embeddings...")
                muril_tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased", cache_dir="./hf_models")
                muril_model = AutoModel.from_pretrained("google/muril-base-cased", cache_dir="./hf_models").to(device)
                muril_model.eval()
                all_txt_emb = get_muril_embeddings(muril_model, muril_tokenizer, df['Text'].astype(str).tolist(), device, batch_size=32)
                save_embeddings(all_txt_emb, text_embedding_path)
            text_embeddings["muril"] = all_txt_emb

    EXPERIMENTS = [
        {
            "encoder": "bge-m3",
            "loss_type": "weighted"
        },
        {
            "encoder": "bge-m3",
            "loss_type": "unweighted"
        },
        {
            "encoder": "muril",
            "loss_type": "weighted"
        },
        {
            "encoder": "muril",
            "loss_type": "unweighted"
        }
    ]

    # trying with multiple seeds - to measure how stable our model is with different random initializations, data shuffling, and train test splits
    seeds = [7, 10, 42, 56, 100]

    all_results = []
    for EXP in EXPERIMENTS:
        TEXT_ENCODER = EXP["encoder"]
        LOSS_TYPE = EXP["loss_type"]

        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {TEXT_ENCODER} + {LOSS_TYPE}")
        print("=" * 80)

        experiment_dir = os.path.join(RESULTS_DIR, TEXT_ENCODER, LOSS_TYPE)
        os.makedirs(experiment_dir, exist_ok=True)

        if TEXT_ENCODER == "bge-m3":
            text_model = SentenceTransformer("BAAI/bge-m3", device=device, cache_folder="./hf_models")
            text_tokenizer = None
        elif TEXT_ENCODER == "muril":
            text_tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased", cache_dir="./hf_models")
            text_model = AutoModel.from_pretrained("google/muril-base-cased", cache_dir="./hf_models").to(device)
            text_model.eval()
        else:
            raise ValueError(f"Unknown encoder: {TEXT_ENCODER}")

        experiment_results = []




        for SEED in seeds:
            print("\n" + "-" * 70)
            print(
                f"Encoder: {TEXT_ENCODER} | "
                f"Loss: {LOSS_TYPE} | "
                f"Seed: {SEED}"
            )
            print("-" * 70)

            set_seed(SEED)
            seed_dir = os.path.join(
                experiment_dir,
                f"seed_{SEED}"
            )

            os.makedirs(
                seed_dir,
                exist_ok=True
            )

            # Train and validation splitting: holding 90% instances for training and remaining 10% for validation
            train_df, val_df = train_test_split(df, test_size=0.10, random_state=SEED, stratify=df['Level1'])

            # Preserve original dataframe indices
            train_indices = train_df.index.to_numpy()
            val_indices = val_df.index.to_numpy()

            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            print("Train:", train_df.shape, " Val:", val_df.shape)

            distribution_plot(df, train_df, val_df, seed_dir)

            # Label encoding the classes for processing
            le1 = LabelEncoder()
            le2 = LabelEncoder()

            train_df['l1_id'] = le1.fit_transform(train_df['Level1'])
            val_df['l1_id'] = le1.transform(val_df['Level1'])

            train_df['l2_id'] = le2.fit_transform(train_df['Level2'])
            val_df['l2_id'] = le2.transform(val_df['Level2'])

            NUM_L1 = len(le1.classes_)
            NUM_L2 = len(le2.classes_)
            print("Level 1 classes:", list(le1.classes_))
            print("Level 2 classes:", list(le2.classes_))

            # Creating embeddings for image and text
            train_img_emb = get_image_embeddings(img_model, train_df['image_path'].tolist(), IMAGE_DIR, batch_size=32)
            val_img_emb = get_image_embeddings(img_model, val_df['image_path'].tolist(), IMAGE_DIR, batch_size=32)

            # train_txt_emb = get_text_embeddings(text_model, train_df['Text'].astype(str).tolist(), batch_size=32)
            # val_txt_emb = get_text_embeddings(text_model, val_df['Text'].astype(str).tolist(), batch_size=32)

            # ============================================================
            # TEXT EMBEDDINGS
            # ============================================================

            if TEXT_ENCODER == "bge-m3":
                train_txt_emb = get_bge_embeddings(
                    text_model,
                    train_df['Text'].astype(str).tolist(),
                    device,
                    batch_size=32
                )
                val_txt_emb = get_bge_embeddings(
                    text_model,
                    val_df['Text'].astype(str).tolist(),
                    device,
                    batch_size=32
                )
            elif TEXT_ENCODER == "muril":
                train_txt_emb = get_muril_embeddings(
                    text_model,
                    text_tokenizer,
                    train_df['Text'].astype(str).tolist(),
                    device,
                    batch_size=32
                )
                val_txt_emb = get_muril_embeddings(
                    text_model,
                    text_tokenizer,
                    val_df['Text'].astype(str).tolist(),
                    device,
                    batch_size=32
                )

                pca = PCA(n_components=512, random_state=SEED)
                train_txt_emb = pca.fit_transform(train_txt_emb)
                val_txt_emb = pca.transform(val_txt_emb)
                explained_variance = pca.explained_variance_ratio_.sum()
                print(f"{TEXT_ENCODER} PCA explained variance: {explained_variance:.4f}")

            # Fusion of Image and Text embeddings
            train_features = np.concatenate([train_img_emb, train_txt_emb], axis=1)
            val_features = np.concatenate([val_img_emb, val_txt_emb], axis=1)
            # print("\nFused feature dim:", train_features.shape[1])
            print("\nImage embedding dimension:", train_img_emb.shape[1])
            print("Text embedding dimension:", train_txt_emb.shape[1])
            print("Fused feature dimension:", train_features.shape[1])

            train_ds = MemeFeatureDataset(train_features, train_df['l1_id'].values, train_df['l2_id'].values)
            val_ds = MemeFeatureDataset(val_features, val_df['l1_id'].values, val_df['l2_id'].values)

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

            input_dim = train_features.shape[1]
            model = HierarchicalClassifier(input_dim, NUM_L1, NUM_L2).to(device)

            model_description(model)

            if LOSS_TYPE == "weighted":
                l1_weights = compute_class_weight(
                    'balanced',
                    classes=np.unique(train_df['l1_id']),
                    y=train_df['l1_id']
                )
                l2_weights = compute_class_weight(
                    'balanced',
                    classes=np.unique(train_df['l2_id']),
                    y=train_df['l2_id']
                )
                l1_weights = torch.tensor(
                    l1_weights,
                    dtype=torch.float32
                ).to(device)
                l2_weights = torch.tensor(
                    l2_weights,
                    dtype=torch.float32
                ).to(device)
                criterion_l1 = nn.CrossEntropyLoss(
                    weight=l1_weights
                )
                criterion_l2 = nn.CrossEntropyLoss(
                    weight=l2_weights
                )
            elif LOSS_TYPE == "unweighted":
                criterion_l1 = nn.CrossEntropyLoss()
                criterion_l2 = nn.CrossEntropyLoss()
            else:
                raise ValueError("LOSS_TYPE must be either 'weighted' or 'unweighted'")
            # need to make it correct
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

            EPOCHS = 60
            def evaluate(loader):
                model.eval()

                total_val_loss = 0

                all_l1_preds, all_l1_true = [], []
                all_l2_preds, all_l2_true = [], []

                with torch.no_grad():
                    for x, y1, y2 in loader:
                        x = x.to(device)
                        y1 = y1.to(device)
                        y2 = y2.to(device)

                        l1_logits, l2_logits = model(x)
                        loss = (criterion_l1(l1_logits, y1) + criterion_l2(l2_logits, y2))
                        total_val_loss += loss.item()
                        all_l1_preds.extend(l1_logits.argmax(1).cpu().numpy())
                        all_l1_true.extend(
                            y1.cpu().numpy()
                        )
                        all_l2_preds.extend(
                            l2_logits.argmax(1).cpu().numpy()
                        )
                        all_l2_true.extend(
                            y2.cpu().numpy()
                        )
                avg_val_loss = total_val_loss / len(loader)
                f1_l1 = f1_score(all_l1_true, all_l1_preds, average='macro')
                f1_l2 = f1_score(all_l2_true, all_l2_preds, average='macro')
                acc_l1 = accuracy_score(all_l1_true, all_l1_preds)
                acc_l2 = accuracy_score(all_l2_true, all_l2_preds)
                return avg_val_loss, f1_l1, f1_l2, acc_l1, acc_l2

            best_f1 = 0
            history = []

            for epoch in range(EPOCHS):
                model.train()
                total_loss = 0
                for x, y1, y2 in train_loader:
                    x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
                    optimizer.zero_grad()
                    l1_logits, l2_logits = model(x)
                    loss = criterion_l1(l1_logits, y1) + criterion_l2(l2_logits, y2)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                # f1_l1, f1_l2, acc_l1, acc_l2 = evaluate(val_loader)
                val_loss, f1_l1, f1_l2, acc_l1, acc_l2 = evaluate(val_loader)

                avg_f1 = (f1_l1 + f1_l2) / 2
                scheduler.step(avg_f1)
                history.append({
                    "epoch": epoch + 1,
                    "loss": total_loss / len(train_loader),
                    "val_loss": val_loss,
                    "val_f1_l1": f1_l1,
                    "val_f1_l2": f1_l2
                })

                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"| train_loss={total_loss / len(train_loader):.4f} "
                    f"| val_loss={val_loss:.4f} "
                    f"| val_F1_L1={f1_l1:.4f} "
                    f"| val_Acc_L1={acc_l1:.4f} "
                    f"| val_F1_L2={f1_l2:.4f} "
                    f"| val_Acc_L2={acc_l2:.4f}"
                )

                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    model_path = os.path.join(
                        seed_dir,
                        f"best_model.pt"
                    )
                    torch.save(model.state_dict(), model_path)
                    print(f"-> saved new best model: {model_path}")

            hist_df = pd.DataFrame(history)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            # Training and Validation Loss
            axes[0].plot(hist_df['epoch'], hist_df['loss'], label='Train Loss')
            axes[0].plot(hist_df['epoch'], hist_df['val_loss'], label='Validation Loss')
            axes[0].set_title('Training vs Validation Loss')
            axes[0].set_xlabel("epoch", fontsize=12)
            axes[0].set_ylabel("loss", fontsize=12)
            axes[0].legend()
            # Validation F1
            axes[1].plot(hist_df['epoch'], hist_df['val_f1_l1'], label='Level 1 F1')
            axes[1].plot(hist_df['epoch'], hist_df['val_f1_l2'], label='Level 2 F1')
            axes[1].legend()
            # axes[1].set_title('Validation Macro-F1')
            axes[1].set_xlabel("epoch", fontsize=12)
            axes[1].set_ylabel("Macro-F1", fontsize=12)
            # fig.suptitle(f"Training/Validation Loss and Validation F1 - seed {SEED}")
            fig.suptitle(f"Validation Macro-F1")
            plt.tight_layout()
            plt.savefig(os.path.join(seed_dir, f"val_macro_f1.png"), dpi=300, bbox_inches='tight')
            # plt.show()

            model.load_state_dict(torch.load(model_path))
            model.eval()

            all_l1_preds, all_l1_true, all_l2_preds, all_l2_true = [], [], [], []
            with torch.no_grad():
                for x, y1, y2 in val_loader:
                    x = x.to(device)
                    l1_logits, l2_logits = model(x)
                    all_l1_preds.extend(l1_logits.argmax(1).cpu().numpy())
                    all_l1_true.extend(y1.numpy())
                    all_l2_preds.extend(l2_logits.argmax(1).cpu().numpy())
                    all_l2_true.extend(y2.numpy())

            # Generate reports
            report_l1 = classification_report(
                all_l1_true,
                all_l1_preds,
                target_names=le1.classes_,
                output_dict=True
            )
            report_l2 = classification_report(
                all_l2_true,
                all_l2_preds,
                target_names=le2.classes_,
                output_dict=True
            )
            # Save reports
            with open(os.path.join(seed_dir, "level1_report.txt"), "w") as f:
                f.write(str(report_l1))
            with open(os.path.join(seed_dir, "level2_report.txt"), "w") as f:
                f.write(str(report_l2))
            # print(report_l1)
            # print(report_l2)

            l1_acc = accuracy_score(all_l1_true, all_l1_preds)
            l2_acc = accuracy_score(all_l2_true, all_l2_preds)
            # Store results for this seed
            experiment_results.append({
                "encoder": TEXT_ENCODER,
                "loss_type": LOSS_TYPE,
                "seed": SEED,
                # Level 1
                "l1_accuracy": l1_acc,
                "l1_macro_f1": report_l1["macro avg"]["f1-score"],
                # "l1_weighted_f1": report_l1["weighted avg"]["f1-score"],
                # Level 2
                "l2_accuracy": l2_acc,
                "l2_macro_f1": report_l2["macro avg"]["f1-score"],
                # "l2_weighted_f1": report_l2["weighted avg"]["f1-score"]
                # Average across Level 1 and Level 2
                "avg_accuracy": (l1_acc + l2_acc) / 2,
                "avg_macro_f1": (report_l1["macro avg"]["f1-score"] + report_l2["macro avg"]["f1-score"]) / 2,
                # "avg_weighted_f1": (report_l1["weighted avg"]["f1-score"] + report_l2["weighted avg"]["f1-score"]) / 2
            })

        # Save result for this config
        experiment_df = pd.DataFrame(experiment_results)
        experiment_df.to_csv(os.path.join(experiment_dir, "seed_comparison.csv"), index=False)
        all_results.extend(experiment_results)

    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(os.path.join(RESULTS_DIR, "all_experiments.csv"), index=False)
    print("\n" + "=" * 100)
    print("ALL EXPERIMENT RESULTS")
    print("=" * 100)
    print(
        all_results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ================================================================
    # FINAL RESULTS: MEAN ± STD FOR EACH CONFIGURATION
    # ================================================================

    metric_columns = [
        "l1_accuracy",
        "l1_macro_f1",
        # "l1_weighted_f1",
        "l2_accuracy",
        "l2_macro_f1",
        # "l2_weighted_f1"
        "avg_accuracy",
        "avg_macro_f1",
        # "avg_weighted_f1"
    ]

    final_results = []

    grouped_results = all_results_df.groupby(
        ["encoder", "loss_type"]
    )

    for (encoder, loss_type), group in grouped_results:
        for metric in metric_columns:
            final_results.append({
                "encoder": encoder,
                "loss_type": loss_type,
                "metric": metric,
                "mean": group[metric].mean(),
                "std": group[metric].std(),
                "min": group[metric].min(),
                "max": group[metric].max()
            })

    final_results_df = pd.DataFrame(
        final_results
    )

    # ================================================================
    # SAVE FINAL RESULTS
    # ================================================================

    final_results_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "final_results.csv"
        ),
        index=False
    )

    # ================================================================
    # PRINT FINAL RESULTS
    # ================================================================

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    print(
        final_results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ================================================================
    # PRINT MEAN ± STD
    # ================================================================

    print("\n" + "=" * 100)
    print("MEAN ± STD")
    print("=" * 100)

    for (encoder, loss_type), group in grouped_results:

        print("\n" + "-" * 70)

        print(
            f"Encoder: {encoder} | "
            f"Loss: {loss_type}"
        )

        print("-" * 70)

        for metric in metric_columns:
            mean = group[metric].mean()
            std = group[metric].std()

            print(
                f"{metric:<25} "
                f"{mean:.4f} ± {std:.4f}"
            )
    # print(all_results_df.to_string(index=False))
    #
    #     metric_columns = [
    #         "l1_accuracy",
    #         "l1_macro_f1",
    #         "l1_weighted_f1",
    #         "l2_accuracy",
    #         "l2_macro_f1",
    #         "l2_weighted_f1"
    #     ]
    #     final_results = []
    #     for metric in metric_columns:
    #         final_results.append({
    #             "encoder": TEXT_ENCODER,
    #             "loss_type": LOSS_TYPE,
    #             "metric": metric,
    #             "mean": results_df[metric].mean(),
    #             "std": results_df[metric].std(),
    #             "min": results_df[metric].min(),
    #             "max": results_df[metric].max()
    #         })
    #     final_results_df = pd.DataFrame(final_results)
    #     final_results_df.to_csv(os.path.join(loss_dir, "final_results.csv"), index=False)
    #
    #     print(
    #         final_results_df.to_string(
    #             index=False,
    #             float_format=lambda x: f"{x:.4f}"
    #         )
    #     )
    #     print("MEAN ± STD")
    #     for _, row in final_results_df.iterrows():
    #         print(
    #             f"{row['metric']:<20} "
    #             f"{row['mean']:.4f} ± {row['std']:.4f}"
    #         )


if __name__ == '__main__':
    main()