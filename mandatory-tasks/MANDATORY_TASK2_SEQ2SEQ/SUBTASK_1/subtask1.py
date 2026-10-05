import pandas as pd
import re
import random
import gc

import torch
import torch.nn as nn

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ============================================================
# 1. LOAD DATA
# ============================================================

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))


# ============================================================
# 2. TOKENIZATION
# ============================================================

def tokenize(sentence):

    sentence = sentence.lower()

    sentence = re.sub(
        r"([?.!,])",
        r" \1 ",
        sentence
    )

    sentence = re.sub(
        r"\s+",
        " ",
        sentence
    ).strip()

    return sentence.split()


# ============================================================
# 3. SPECIAL TOKENS
# ============================================================

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN
]


# ============================================================
# 4. BUILD VOCABULARY
# ============================================================

def build_vocab(sentences, min_freq=2):

    counter = Counter()

    for sentence in sentences:

        tokens = tokenize(sentence)

        for token in tokens:
            counter[token] += 1

    word_to_idx = {}

    for idx, token in enumerate(SPECIAL_TOKENS):
        word_to_idx[token] = idx

    for word, frequency in counter.items():

        if frequency >= min_freq:

            if word not in word_to_idx:

                word_to_idx[word] = len(word_to_idx)

    idx_to_word = {
        idx: word
        for word, idx in word_to_idx.items()
    }

    return word_to_idx, idx_to_word


en_word_to_idx, en_idx_to_word = build_vocab(
    train_df["en"]
)

fr_word_to_idx, fr_idx_to_word = build_vocab(
    train_df["fr"]
)

print("English vocabulary:", len(en_word_to_idx))
print("French vocabulary:", len(fr_word_to_idx))


# ============================================================
# 5. SENTENCE TO IDS
# ============================================================

def sentence_to_ids(sentence, word_to_idx):

    tokens = tokenize(sentence)

    tokens = [
        SOS_TOKEN
    ] + tokens + [
        EOS_TOKEN
    ]

    ids = []

    for token in tokens:

        if token in word_to_idx:

            ids.append(
                word_to_idx[token]
            )

        else:

            ids.append(
                word_to_idx[UNK_TOKEN]
            )

    return ids


# ============================================================
# 6. DATASET
# ============================================================

class TranslationDataset(Dataset):

    def __init__(
        self,
        dataframe,
        src_vocab,
        trg_vocab
    ):

        self.dataframe = dataframe
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        source_sentence = self.dataframe.iloc[index]["en"]

        target_sentence = self.dataframe.iloc[index]["fr"]

        source_ids = sentence_to_ids(
            source_sentence,
            self.src_vocab
        )

        target_ids = sentence_to_ids(
            target_sentence,
            self.trg_vocab
        )

        return (
            torch.tensor(
                source_ids,
                dtype=torch.long
            ),
            torch.tensor(
                target_ids,
                dtype=torch.long
            )
        )


# ============================================================
# 7. SMALL TRAINING SET
# ============================================================

TRAIN_SIZE = 20000

small_train_df = train_df.iloc[
    :TRAIN_SIZE
].copy()

train_dataset = TranslationDataset(
    small_train_df,
    en_word_to_idx,
    fr_word_to_idx
)

val_dataset = TranslationDataset(
    val_df,
    en_word_to_idx,
    fr_word_to_idx
)

test_dataset = TranslationDataset(
    test_df,
    en_word_to_idx,
    fr_word_to_idx
)


# ============================================================
# 8. COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    source_batch = []

    target_batch = []

    for source, target in batch:

        source_batch.append(source)

        target_batch.append(target)

    source_batch = pad_sequence(
        source_batch,
        batch_first=True,
        padding_value=en_word_to_idx[PAD_TOKEN]
    )

    target_batch = pad_sequence(
        target_batch,
        batch_first=True,
        padding_value=fr_word_to_idx[PAD_TOKEN]
    )

    return source_batch, target_batch


# ============================================================
# 9. DATALOADERS
# ============================================================

BATCH_SIZE = 8

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# ============================================================
# 10. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 11. ENCODER
# ============================================================

class Encoder(nn.Module):

    def __init__(
        self,
        input_size,
        embedding_dim,
        hidden_dim
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            input_size,
            embedding_dim,
            padding_idx=en_word_to_idx[PAD_TOKEN]
        )

        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

    def forward(self, source):

        embedded = self.embedding(source)

        outputs, hidden = self.gru(
            embedded
        )

        return hidden


# ============================================================
# 12. DECODER
# ============================================================

class Decoder(nn.Module):

    def __init__(
        self,
        output_size,
        embedding_dim,
        hidden_dim
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            output_size,
            embedding_dim,
            padding_idx=fr_word_to_idx[PAD_TOKEN]
        )

        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc_out = nn.Linear(
            hidden_dim,
            output_size
        )

    def forward(
        self,
        input_token,
        hidden
    ):

        input_token = input_token.unsqueeze(1)

        embedded = self.embedding(
            input_token
        )

        output, hidden = self.gru(
            embedded,
            hidden
        )

        prediction = self.fc_out(
            output.squeeze(1)
        )

        return prediction, hidden


# ============================================================
# 13. MODEL SETTINGS
# ============================================================

INPUT_DIM = len(en_word_to_idx)

OUTPUT_DIM = len(fr_word_to_idx)

ENC_EMB_DIM = 128

HIDDEN_DIM = 256


encoder = Encoder(
    input_size=INPUT_DIM,
    embedding_dim=ENC_EMB_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)


decoder = Decoder(
    output_size=OUTPUT_DIM,
    embedding_dim=ENC_EMB_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)


# ============================================================
# 14. SEQ2SEQ
# ============================================================

class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        device
    ):

        super().__init__()

        self.encoder = encoder

        self.decoder = decoder

        self.device = device

    def forward(
        self,
        source,
        target,
        teacher_forcing_ratio=0.5
    ):

        target_length = target.size(1)

        hidden = self.encoder(
            source
        )

        input_token = target[:, 0]

        outputs = []

        for t in range(
            1,
            target_length
        ):

            output, hidden = self.decoder(
                input_token,
                hidden
            )

            outputs.append(output)

            best_guess = output.argmax(
                dim=1
            )

            use_teacher_forcing = (
                random.random()
                < teacher_forcing_ratio
            )

            if use_teacher_forcing:

                input_token = target[:, t]

            else:

                input_token = best_guess

        outputs = torch.stack(
            outputs,
            dim=1
        )

        return outputs


model = Seq2Seq(
    encoder,
    decoder,
    device
).to(device)


# ============================================================
# 15. LOSS AND OPTIMIZER
# ============================================================

PAD_IDX = fr_word_to_idx[PAD_TOKEN]

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 16. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    clip=1
):

    model.train()

    total_loss = 0

    for batch_idx, (source, target) in enumerate(
        data_loader
    ):

        source = source.to(device)

        target = target.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            source,
            target,
            teacher_forcing_ratio=0.5
        )

        output_dim = output.shape[-1]

        target_for_loss = target[
            :,
            1:1 + output.size(1)
        ]

        output_for_loss = output.reshape(
            -1,
            output_dim
        )

        target_for_loss = target_for_loss.reshape(
            -1
        )

        loss = criterion(
            output_for_loss,
            target_for_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            clip
        )

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 250 == 0:

            print(
                f"Batch {batch_idx}/{len(data_loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    return total_loss / len(data_loader)


# ============================================================
# 17. VALIDATION FUNCTION
# ============================================================

def evaluate(
    model,
    data_loader,
    criterion
):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for source, target in data_loader:

            source = source.to(device)

            target = target.to(device)

            output = model(
                source,
                target,
                teacher_forcing_ratio=0
            )

            output_dim = output.shape[-1]

            target_for_loss = target[
                :,
                1:1 + output.size(1)
            ]

            output_for_loss = output.reshape(
                -1,
                output_dim
            )

            target_for_loss = target_for_loss.reshape(
                -1
            )

            loss = criterion(
                output_for_loss,
                target_for_loss
            )

            total_loss += loss.item()

    return total_loss / len(data_loader)


# ============================================================
# 18. TRAINING
# ============================================================

NUM_EPOCHS = 3

best_val_loss = float("inf")

patience = 2

epochs_without_improvement = 0


for epoch in range(
    NUM_EPOCHS
):

    print()
    print(
        f"========== EPOCH {epoch + 1} =========="
    )

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )

    print(
        f"Training Loss: {train_loss:.4f}"
    )

    val_loss = evaluate(
        model,
        val_loader,
        criterion
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            "best_seq2seq_model.pt"
        )

        print(
            "Best model saved."
        )

    else:

        epochs_without_improvement += 1

        print(
            "Validation loss did not improve."
        )

    if epochs_without_improvement >= patience:

        print(
            "Early stopping."
        )

        break

    if device.type == "cuda":

        torch.cuda.empty_cache()

    gc.collect()


print()
print(
    "Training complete."
)
print(
    "Best validation loss:",
    best_val_loss
)