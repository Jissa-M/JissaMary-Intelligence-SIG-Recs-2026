import json
import glob
import os
import re
import pickle

from collections import Counter
from datasets import load_dataset


# ==========================================
# 1. LOAD DATASET
# ==========================================

dataset = load_dataset("festvox/cmu_hinglish_dog")

print(dataset)
print(dataset["train"].column_names)


# ==========================================
# 2. INSPECT DATASET
# ==========================================

row = dataset["train"][0]

print("docIdx:", row["docIdx"])
print("wikiDocumentIdx:", row["wikiDocumentIdx"])
print("uid:", row["uid"])
print("user2_id:", row["user2_id"])

print("\nEnglish:")
print(row["translation"]["en"])

print("\nHinglish:")
print(row["translation"]["hi_en"])


# ==========================================
# 3. LOAD WIKIDATA DOCUMENTS
# ==========================================

wiki_docs = {}

for path in glob.glob("WikiData/*.json"):

    with open(path, "r", encoding="utf-8") as f:
        document = json.load(f)

    wiki_id = document["wikiDocumentIdx"]

    wiki_docs[wiki_id] = document

print("Number of Wiki documents:", len(wiki_docs))


# ==========================================
# 4. TEST WIKIDATA
# ==========================================

print(wiki_docs[14].keys())
print(wiki_docs[14]["0"])
print(wiki_docs[14]["1"])


# ==========================================
# 5. CHECK DATASET → WIKIDATA MAPPING
# ==========================================

row = dataset["train"][0]

wiki_id = row["wikiDocumentIdx"]
doc_idx = str(row["docIdx"])

document = wiki_docs[wiki_id][doc_idx]

print("Wiki ID:", wiki_id)
print("Document section:", doc_idx)
print("\nDocument:")
print(document)


# ==========================================
# 6. CHECK CONVERSATION BOUNDARIES
# ==========================================

previous_user2 = None

for i in range(len(dataset["train"])):

    row = dataset["train"][i]

    if row["user2_id"] != previous_user2:

        print(
            "\nNEW POSSIBLE CONVERSATION AT ROW:",
            i,
            "| user2_id:", row["user2_id"],
            "| wiki:", row["wikiDocumentIdx"],
            "| docIdx:", row["docIdx"]
        )

    previous_user2 = row["user2_id"]


# ==========================================
# 7. CLEAN TEXT
# ==========================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# 8. GET DOCUMENT
# ==========================================

def get_document(row):

    wiki_id = row["wikiDocumentIdx"]
    doc_idx = str(row["docIdx"])

    if wiki_id not in wiki_docs:
        return ""

    document = wiki_docs[wiki_id]

    if doc_idx not in document:
        return ""

    section = document[doc_idx]

    if isinstance(section, dict):

        parts = []

        for key, value in section.items():

            if isinstance(value, list):
                value = " ".join(str(x) for x in value)

            parts.append(str(value))

        section = " ".join(parts)

    return clean_text(section)


# ==========================================
# 9. TEST DOCUMENT EXTRACTION
# ==========================================

row = dataset["train"][5]

print("Hinglish:")
print(row["translation"]["hi_en"])

print("\nDocument:")
print(get_document(row))


# ==========================================
# 10. BUILD HISTORY + DOCUMENT + TARGET
# ==========================================

def build_examples(split):

    examples = []

    current_conversation = None
    history = []

    for i in range(len(dataset[split])):

        row = dataset[split][i]

        conversation_id = row["user2_id"]

        if conversation_id != current_conversation:

            current_conversation = conversation_id
            history = []

        target = clean_text(row["translation"]["hi_en"])

        if target == "":
            continue

        document = get_document(row)

        if len(history) > 0:

            history_text = " ".join(history)

            examples.append({
                "history": history_text,
                "document": document,
                "target": target,
                "wikiDocumentIdx": row["wikiDocumentIdx"],
                "docIdx": row["docIdx"]
            })

        speaker = row["uid"]

        if speaker == "user1":
            history.append("<USR1> " + target)

        else:
            history.append("<USR2> " + target)

    return examples


train_examples = build_examples("train")
val_examples = build_examples("validation")
test_examples = build_examples("test")


print("Train examples:", len(train_examples))
print("Validation examples:", len(val_examples))
print("Test examples:", len(test_examples))


# ==========================================
# 11. INSPECT EXAMPLES
# ==========================================

for i in range(5):

    example = train_examples[i]

    print("\n" + "=" * 80)

    print("\nHISTORY:")
    print(example["history"])

    print("\nDOCUMENT:")
    print(example["document"][:500])

    print("\nTARGET:")
    print(example["target"])


# ==========================================
# 12. LENGTH ANALYSIS
# ==========================================

def word_count(text):
    return len(text.split())


train_history_lengths = []
train_document_lengths = []
train_target_lengths = []

for example in train_examples:

    train_history_lengths.append(
        word_count(example["history"])
    )

    train_document_lengths.append(
        word_count(example["document"])
    )

    train_target_lengths.append(
        word_count(example["target"])
    )


print(
    "Average history length:",
    sum(train_history_lengths) / len(train_history_lengths)
)

print(
    "Average document length:",
    sum(train_document_lengths) / len(train_document_lengths)
)

print(
    "Average target length:",
    sum(train_target_lengths) / len(train_target_lengths)
)

print(
    "Maximum history length:",
    max(train_history_lengths)
)

print(
    "Maximum document length:",
    max(train_document_lengths)
)

print(
    "Maximum target length:",
    max(train_target_lengths)
)


# ==========================================
# 13. SAVE PROCESSED DATA
# ==========================================

with open("task3_train_examples.pkl", "wb") as f:
    pickle.dump(train_examples, f)

with open("task3_val_examples.pkl", "wb") as f:
    pickle.dump(val_examples, f)

with open("task3_test_examples.pkl", "wb") as f:
    pickle.dump(test_examples, f)

print("Saved Task 3 processed datasets.")


# ==========================================
# 14. TOKENIZATION + VOCABULARY
# ==========================================

MAX_HISTORY_LEN = 80
MAX_DOCUMENT_LEN = 150
MAX_TARGET_LEN = 40


def tokenize(text):

    text = text.lower()

    # Preserve speaker identity
    text = text.replace("<usr1>", " usr1speaker ")
    text = text.replace("<usr2>", " usr2speaker ")

    tokens = re.findall(
        r"\w+|[^\w\s]",
        text,
        re.UNICODE
    )

    return tokens


def limit_tokens(tokens, max_len):
    return tokens[:max_len]


# ==========================================
# 15. BUILD VOCABULARY
# ==========================================

counter = Counter()

for example in train_examples:

    history_tokens = limit_tokens(
        tokenize(example["history"]),
        MAX_HISTORY_LEN
    )

    document_tokens = limit_tokens(
        tokenize(example["document"]),
        MAX_DOCUMENT_LEN
    )

    target_tokens = limit_tokens(
        tokenize(example["target"]),
        MAX_TARGET_LEN
    )

    counter.update(history_tokens)
    counter.update(document_tokens)
    counter.update(target_tokens)


# ==========================================
# 16. SPECIAL TOKENS
# ==========================================

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

special_tokens = [
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN
]


# ==========================================
# 17. CREATE VOCABULARY
# ==========================================

MIN_FREQ = 2

vocab = special_tokens.copy()

for word, frequency in counter.items():

    if frequency >= MIN_FREQ:
        vocab.append(word)


vocab = list(dict.fromkeys(vocab))


# ==========================================
# 18. WORD ↔ ID
# ==========================================

word_to_idx = {}

for idx, word in enumerate(vocab):
    word_to_idx[word] = idx


idx_to_word = {}

for word, idx in word_to_idx.items():
    idx_to_word[idx] = word


print("\nVocabulary size:", len(vocab))
print("PAD:", word_to_idx[PAD_TOKEN])
print("SOS:", word_to_idx[SOS_TOKEN])
print("EOS:", word_to_idx[EOS_TOKEN])
print("UNK:", word_to_idx[UNK_TOKEN])


# ==========================================
# 19. TEST TOKENIZATION
# ==========================================

example = train_examples[0]

print("\nOriginal:")
print(example["history"])

print("\nTokens:")
print(tokenize(example["history"])[:30])

print("\nDocument tokens:")
print(tokenize(example["document"])[:30])

print("\nTarget tokens:")
print(tokenize(example["target"]))


print(
    "\nHistory:",
    len(
        limit_tokens(
            tokenize(example["history"]),
            MAX_HISTORY_LEN
        )
    )
)

print(
    "Document:",
    len(
        limit_tokens(
            tokenize(example["document"]),
            MAX_DOCUMENT_LEN
        )
    )
)

print(
    "Target:",
    len(
        limit_tokens(
            tokenize(example["target"]),
            MAX_TARGET_LEN
        )
    )
)

# ==========================================
# TASK 3 - DATASET + DATALOADER
# ==========================================

import torch
from torch.utils.data import Dataset, DataLoader


# ==========================================
# 1. SELECT DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# 2. CONVERT TOKENS TO IDs
# ==========================================

def tokens_to_ids(tokens):

    ids = []

    for token in tokens:

        if token in word_to_idx:
            ids.append(word_to_idx[token])
        else:
            ids.append(word_to_idx[UNK_TOKEN])

    return ids


# ==========================================
# 3. PAD SEQUENCE
# ==========================================

def pad_sequence(ids, max_len):

    ids = ids[:max_len]

    while len(ids) < max_len:
        ids.append(word_to_idx[PAD_TOKEN])

    return ids


# ==========================================
# 4. PROCESS ONE EXAMPLE
# ==========================================

def process_example(example):

    history_tokens = tokenize(example["history"])
    document_tokens = tokenize(example["document"])
    target_tokens = tokenize(example["target"])

    history_tokens = limit_tokens(
        history_tokens,
        MAX_HISTORY_LEN
    )

    document_tokens = limit_tokens(
        document_tokens,
        MAX_DOCUMENT_LEN
    )

    target_tokens = limit_tokens(
        target_tokens,
        MAX_TARGET_LEN - 2
    )

    # History
    history_ids = tokens_to_ids(history_tokens)
    history_ids = pad_sequence(
        history_ids,
        MAX_HISTORY_LEN
    )

    # Document
    document_ids = tokens_to_ids(document_tokens)
    document_ids = pad_sequence(
        document_ids,
        MAX_DOCUMENT_LEN
    )

    # Target input
    target_input_tokens = (
        [SOS_TOKEN] +
        target_tokens
    )

    target_input_ids = tokens_to_ids(
        target_input_tokens
    )

    target_input_ids = pad_sequence(
        target_input_ids,
        MAX_TARGET_LEN
    )

    # Target output
    target_output_tokens = (
        target_tokens +
        [EOS_TOKEN]
    )

    target_output_ids = tokens_to_ids(
        target_output_tokens
    )

    target_output_ids = pad_sequence(
        target_output_ids,
        MAX_TARGET_LEN
    )

    return (
        history_ids,
        document_ids,
        target_input_ids,
        target_output_ids
    )


# ==========================================
# 5. PYTORCH DATASET
# ==========================================

class DialogueDataset(Dataset):

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        example = self.examples[index]

        history_ids, document_ids, target_input_ids, target_output_ids = process_example(
            example
        )

        return (
            torch.tensor(history_ids, dtype=torch.long),
            torch.tensor(document_ids, dtype=torch.long),
            torch.tensor(target_input_ids, dtype=torch.long),
            torch.tensor(target_output_ids, dtype=torch.long)
        )


# ==========================================
# 6. CREATE DATASETS
# ==========================================

train_dataset = DialogueDataset(train_examples)
val_dataset = DialogueDataset(val_examples)
test_dataset = DialogueDataset(test_examples)


# ==========================================
# 7. CREATE DATALOADERS
# ==========================================

BATCH_SIZE = 8

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 8. TEST ONE BATCH
# ==========================================

history_batch, document_batch, target_input_batch, target_output_batch = next(
    iter(train_loader)
)

print("History shape:", history_batch.shape)
print("Document shape:", document_batch.shape)
print("Target input shape:", target_input_batch.shape)
print("Target output shape:", target_output_batch.shape)

# ==========================================
# TASK 3 - BASELINE MODEL
# History -> Attention -> Decoder
# ==========================================

import torch
import torch.nn as nn
import random

EMBED_SIZE = 64
HIDDEN_SIZE = 128

class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=word_to_idx[PAD_TOKEN]
        )

        self.gru = nn.GRU(
            embed_size,
            hidden_size,
            batch_first=True
        )

    def forward(self, x):

        embedded = self.embedding(x)

        outputs, hidden = self.gru(embedded)

        return outputs, hidden


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.attention = nn.Linear(
            hidden_size * 2,
            hidden_size
        )

        self.score = nn.Linear(
            hidden_size,
            1,
            bias=False
        )

    def forward(self, decoder_hidden, encoder_outputs, mask):

        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        decoder_hidden = decoder_hidden[-1]

        decoder_hidden = decoder_hidden.unsqueeze(1)

        decoder_hidden = decoder_hidden.repeat(
            1,
            seq_len,
            1
        )

        combined = torch.cat(
            (decoder_hidden, encoder_outputs),
            dim=2
        )

        energy = torch.tanh(
            self.attention(combined)
        )

        scores = self.score(energy).squeeze(2)

        scores = scores.masked_fill(
            mask == 0,
            -1e10
        )

        attention_weights = torch.softmax(
            scores,
            dim=1
        )

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        )

        return context, attention_weights


class Decoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        attention
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=word_to_idx[PAD_TOKEN]
        )

        self.attention = attention

        self.gru = nn.GRU(
            embed_size + hidden_size,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size * 2,
            vocab_size
        )

    def forward(
        self,
        input_token,
        hidden,
        encoder_outputs,
        mask
    ):

        embedded = self.embedding(
            input_token
        )

        embedded = embedded.unsqueeze(1)

        context, attention_weights = self.attention(
            hidden,
            encoder_outputs,
            mask
        )

        gru_input = torch.cat(
            (embedded, context),
            dim=2
        )

        output, hidden = self.gru(
            gru_input,
            hidden
        )

        output = output.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc(
            torch.cat((output, context), dim=1)
        )

        return prediction, hidden, attention_weights


class BaselineSeq2Seq(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        pad_idx
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size,
            embed_size,
            hidden_size
        )

        self.attention = Attention(
            hidden_size
        )

        self.decoder = Decoder(
            vocab_size,
            embed_size,
            hidden_size,
            self.attention
        )

        self.pad_idx = pad_idx

    def forward(
        self,
        source,
        target,
        teacher_forcing_ratio=0.5
    ):

        batch_size = source.size(0)
        target_len = target.size(1)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(
            batch_size,
            target_len,
            vocab_size,
            device=source.device
        )

        encoder_outputs, hidden = self.encoder(source)

        mask = source != self.pad_idx

        input_token = target[:, 0]

        for t in range(1, target_len):

            prediction, hidden, _ = self.decoder(
                input_token,
                hidden,
                encoder_outputs,
                mask
            )

            outputs[:, t] = prediction

            teacher_force = random.random() < teacher_forcing_ratio

            predicted_token = prediction.argmax(1)

            if teacher_force:
                input_token = target[:, t]
            else:
                input_token = predicted_token

        return outputs


vocab_size = len(word_to_idx)

baseline_model = BaselineSeq2Seq(
    vocab_size,
    EMBED_SIZE,
    HIDDEN_SIZE,
    word_to_idx[PAD_TOKEN]
).to(device)

print("Vocabulary size:", vocab_size)
print("Baseline model created.")

# ==========================================
# TASK 3 - TRAIN BASELINE
# ==========================================

import torch.optim as optim

optimizer = optim.Adam(
    baseline_model.parameters(),
    lr=0.001
)

criterion = nn.CrossEntropyLoss(
    ignore_index=word_to_idx[PAD_TOKEN]
)

N_EPOCHS = 1
CLIP = 1

for epoch in range(N_EPOCHS):

    baseline_model.train()

    total_loss = 0

    for batch in train_loader:

        history = batch[0].to(device)
        target_input = batch[2].to(device)
        target_output = batch[3].to(device)

        optimizer.zero_grad()

        output = baseline_model(
            history,
            target_input,
            teacher_forcing_ratio=0.5
        )

        output_dim = output.size(-1)

        output = output[:, 1:].reshape(
            -1,
            output_dim
        )

        target_output = target_output[:, 1:].reshape(-1)

        loss = criterion(
            output,
            target_output
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            baseline_model.parameters(),
            CLIP
        )

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{N_EPOCHS} | "
        f"Train Loss: {average_loss:.4f}"
    )
    
# ==========================================
# TASK 3 - BASELINE EVALUATION
# ==========================================

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

idx_to_word = {
    idx: word
    for word, idx in word_to_idx.items()
}

def decode_ids(ids):
    words = []

    for idx in ids:
        word = idx_to_word.get(idx, UNK_TOKEN)

        if word == EOS_TOKEN:
            break

        if word not in [PAD_TOKEN, SOS_TOKEN]:
            words.append(word)

    return words


def generate_baseline(source, max_len=MAX_TARGET_LEN):

    baseline_model.eval()

    with torch.no_grad():

        encoder_outputs, hidden = baseline_model.encoder(source)

        mask = source != word_to_idx[PAD_TOKEN]

        input_token = torch.tensor(
            [word_to_idx[SOS_TOKEN]],
            device=device
        )

        generated = []

        for _ in range(max_len):

            prediction, hidden, _ = baseline_model.decoder(
                input_token,
                hidden,
                encoder_outputs,
                mask
            )

            predicted_token = prediction.argmax(1)

            token_id = predicted_token.item()

            if token_id == word_to_idx[EOS_TOKEN]:
                break

            if token_id not in [
                word_to_idx[PAD_TOKEN],
                word_to_idx[SOS_TOKEN]
            ]:
                generated.append(token_id)

            input_token = predicted_token

    return generated


references = []
hypotheses = []

test_examples_count = 0

for batch in test_loader:

    history = batch[0].to(device)
    target_output = batch[3].to(device)

    for i in range(history.size(0)):

        prediction = generate_baseline(
            history[i:i+1]
        )

        reference = decode_ids(
            target_output[i].tolist()
        )

        hypothesis = decode_ids(
            prediction
        )

        if len(reference) > 0:
            references.append([reference])
            hypotheses.append(hypothesis)

        test_examples_count += 1


smooth = SmoothingFunction().method1

bleu = corpus_bleu(
    references,
    hypotheses,
    smoothing_function=smooth
)

print("Test examples:", test_examples_count)
print("Baseline BLEU:", bleu)


# ==========================================
# TASK 3 - GROUNDED MODEL
# History + Document -> Hinglish Response
# ==========================================

class GroundedSeq2Seq(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        pad_idx
    ):
        super().__init__()

        self.pad_idx = pad_idx

        # Shared embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=pad_idx
        )

        # History encoder
        self.history_encoder = nn.GRU(
            embed_size,
            hidden_size,
            batch_first=True
        )

        # Document encoder
        self.document_encoder = nn.GRU(
            embed_size,
            hidden_size,
            batch_first=True
        )

        # Attention over history
        self.history_attention = Attention(hidden_size)

        # Attention over document
        self.document_attention = Attention(hidden_size)

        # Decoder
        self.decoder_gru = nn.GRU(
            embed_size + hidden_size * 2,
            hidden_size,
            batch_first=True
        )

        # Final prediction layer
        self.fc = nn.Linear(
            hidden_size * 3,
            vocab_size
        )

    def forward(
        self,
        history,
        document,
        target,
        teacher_forcing_ratio=0.5
    ):

        batch_size = history.size(0)
        target_len = target.size(1)

        # -----------------------------
        # Encode history
        # -----------------------------

        history_embedded = self.embedding(history)

        history_outputs, history_hidden = self.history_encoder(
            history_embedded
        )

        # -----------------------------
        # Encode document
        # -----------------------------

        document_embedded = self.embedding(document)

        document_outputs, document_hidden = self.document_encoder(
            document_embedded
        )

        # Start decoder with history hidden state
        hidden = history_hidden

        history_mask = history != self.pad_idx
        document_mask = document != self.pad_idx

        outputs = torch.zeros(
            batch_size,
            target_len,
            self.fc.out_features,
            device=history.device
        )

        input_token = target[:, 0]

        # -----------------------------
        # Decode
        # -----------------------------

        for t in range(1, target_len):

            history_context, _ = self.history_attention(
                hidden,
                history_outputs,
                history_mask
            )

            document_context, _ = self.document_attention(
                hidden,
                document_outputs,
                document_mask
            )

            embedded = self.embedding(
                input_token
            ).unsqueeze(1)

            decoder_input = torch.cat(
                (
                    embedded,
                    history_context,
                    document_context
                ),
                dim=2
            )

            decoder_output, hidden = self.decoder_gru(
                decoder_input,
                hidden
            )

            decoder_output = decoder_output.squeeze(1)
            history_context = history_context.squeeze(1)
            document_context = document_context.squeeze(1)

            prediction = self.fc(
                torch.cat(
                    (
                        decoder_output,
                        history_context,
                        document_context
                    ),
                    dim=1
                )
            )

            outputs[:, t] = prediction

            teacher_force = (
                random.random() < teacher_forcing_ratio
            )

            predicted_token = prediction.argmax(1)

            if teacher_force:
                input_token = target[:, t]
            else:
                input_token = predicted_token

        return outputs


grounded_model = GroundedSeq2Seq(
    vocab_size=len(word_to_idx),
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    pad_idx=word_to_idx[PAD_TOKEN]
).to(device)

print("Grounded model created.")
print("Parameters:",
      sum(p.numel() for p in grounded_model.parameters()))

# ==========================================
# TASK 3 - TRAIN GROUNDED MODEL
# ==========================================

optimizer = optim.Adam(
    grounded_model.parameters(),
    lr=0.001
)

criterion = nn.CrossEntropyLoss(
    ignore_index=word_to_idx[PAD_TOKEN]
)

N_EPOCHS = 1
CLIP = 1

for epoch in range(N_EPOCHS):

    grounded_model.train()

    total_loss = 0

    for batch in train_loader:

        history = batch[0].to(device)
        document = batch[1].to(device)
        target_input = batch[2].to(device)
        target_output = batch[3].to(device)

        optimizer.zero_grad()

        output = grounded_model(
            history,
            document,
            target_input,
            teacher_forcing_ratio=0.5
        )

        output_dim = output.size(-1)

        output = output[:, 1:].reshape(
            -1,
            output_dim
        )

        target_output = target_output[:, 1:].reshape(-1)

        loss = criterion(
            output,
            target_output
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            grounded_model.parameters(),
            CLIP
        )

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{N_EPOCHS} | "
        f"Grounded Train Loss: {average_loss:.4f}"
    )
    
# ==========================================
# TASK 3 - GROUNDED EVALUATION
# ==========================================

def generate_grounded(history, document, max_len=MAX_TARGET_LEN):

    grounded_model.eval()

    with torch.no_grad():

        history_embedded = grounded_model.embedding(history)
        document_embedded = grounded_model.embedding(document)

        history_outputs, hidden = grounded_model.history_encoder(
            history_embedded
        )

        document_outputs, _ = grounded_model.document_encoder(
            document_embedded
        )

        history_mask = history != word_to_idx[PAD_TOKEN]
        document_mask = document != word_to_idx[PAD_TOKEN]

        input_token = torch.tensor(
            [word_to_idx[SOS_TOKEN]],
            device=device
        )

        generated = []

        for _ in range(max_len):

            history_context, _ = grounded_model.history_attention(
                hidden,
                history_outputs,
                history_mask
            )

            document_context, _ = grounded_model.document_attention(
                hidden,
                document_outputs,
                document_mask
            )

            embedded = grounded_model.embedding(
                input_token
            ).unsqueeze(1)

            decoder_input = torch.cat(
                (
                    embedded,
                    history_context,
                    document_context
                ),
                dim=2
            )

            decoder_output, hidden = grounded_model.decoder_gru(
                decoder_input,
                hidden
            )

            decoder_output = decoder_output.squeeze(1)
            history_context = history_context.squeeze(1)
            document_context = document_context.squeeze(1)

            prediction = grounded_model.fc(
                torch.cat(
                    (
                        decoder_output,
                        history_context,
                        document_context
                    ),
                    dim=1
                )
            )

            predicted_token = prediction.argmax(1)
            token_id = predicted_token.item()

            if token_id == word_to_idx[EOS_TOKEN]:
                break

            if token_id not in [
                word_to_idx[PAD_TOKEN],
                word_to_idx[SOS_TOKEN]
            ]:
                generated.append(token_id)

            input_token = predicted_token

    return generated


references = []
hypotheses = []

for batch in test_loader:

    history = batch[0].to(device)
    document = batch[1].to(device)
    target_output = batch[3].to(device)

    for i in range(history.size(0)):

        prediction = generate_grounded(
            history[i:i+1],
            document[i:i+1]
        )

        reference = decode_ids(
            target_output[i].tolist()
        )

        hypothesis = decode_ids(
            prediction
        )

        if len(reference) > 0:
            references.append([reference])
            hypotheses.append(hypothesis)


smooth = SmoothingFunction().method1

grounded_bleu = corpus_bleu(
    references,
    hypotheses,
    smoothing_function=smooth
)

print("Test examples:", len(references))
print("Grounded BLEU:", grounded_bleu)

# ==========================================
# FINAL QUALITATIVE COMPARISON
# ==========================================

grounded_model.eval()
baseline_model.eval()

shown = 0

for batch in test_loader:

    history = batch[0].to(device)
    document = batch[1].to(device)
    target_output = batch[3].to(device)

    for i in range(history.size(0)):

        baseline_ids = generate_baseline(history[i:i+1])
        grounded_ids = generate_grounded(
            history[i:i+1],
            document[i:i+1]
        )

        actual = decode_ids(target_output[i].tolist())

        print("\n-----------------------------")
        print("Actual   :", " ".join(actual))
        print("Baseline :", " ".join(decode_ids(baseline_ids)))
        print("Grounded :", " ".join(decode_ids(grounded_ids)))

        shown += 1

        if shown == 5:
            break

    if shown == 5:
        break
torch.save(
    baseline_model.state_dict(),
    "task3_baseline.pt"
)

torch.save(
    grounded_model.state_dict(),
    "task3_grounded.pt"
)

print("Models saved.")