import pandas as pd
import re
import random
import gc

import torch
import torch.nn as nn

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu


# ============================================================
# 1. LOAD DATA
# ============================================================

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

TRAIN_SIZE = 20000

train_subset = train_df.iloc[:TRAIN_SIZE].copy()

print("Training samples:", len(train_subset))
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

print(
    "English vocabulary:",
    len(en_word_to_idx)
)

print(
    "French vocabulary:",
    len(fr_word_to_idx)
)


# ============================================================
# 5. SENTENCE TO IDS
# ============================================================

def sentence_to_ids(sentence, word_to_idx):

    tokens = tokenize(sentence)

    tokens = (
        [SOS_TOKEN]
        + tokens
        + [EOS_TOKEN]
    )

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

        source_sentence = (
            self.dataframe.iloc[index]["en"]
        )

        target_sentence = (
            self.dataframe.iloc[index]["fr"]
        )

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
# 7. CREATE DATASETS
# ============================================================

train_dataset = TranslationDataset(
    train_subset,
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
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True
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
# 11. ATTENTION ENCODER
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

        return outputs, hidden


# ============================================================
# 12. BAHDANAU ATTENTION
# ============================================================

class BahdanauAttention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attn = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.v = nn.Linear(
            hidden_dim,
            1,
            bias=False
        )

    def forward(
        self,
        hidden,
        encoder_outputs,
        mask=None
    ):

        hidden = hidden[-1].unsqueeze(1)

        hidden = hidden.repeat(
            1,
            encoder_outputs.size(1),
            1
        )

        energy = torch.tanh(
            self.attn(
                torch.cat(
                    (
                        hidden,
                        encoder_outputs
                    ),
                    dim=2
                )
            )
        )

        attention = self.v(
            energy
        ).squeeze(2)

        if mask is not None:

            attention = attention.masked_fill(
                mask == 0,
                -1e10
            )

        attention = torch.softmax(
            attention,
            dim=1
        )

        return attention


# ============================================================
# 13. ATTENTION DECODER
# ============================================================

class AttentionDecoder(nn.Module):

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

        self.attention = BahdanauAttention(
            hidden_dim
        )

        self.gru = nn.GRU(
            embedding_dim + hidden_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc_out = nn.Linear(
            hidden_dim * 2 + embedding_dim,
            output_size
        )

    def forward(
        self,
        input_token,
        hidden,
        encoder_outputs,
        mask
    ):

        input_token = input_token.unsqueeze(1)

        embedded = self.embedding(
            input_token
        )

        attention_weights = self.attention(
            hidden,
            encoder_outputs,
            mask
        )

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        )

        rnn_input = torch.cat(
            (
                embedded,
                context
            ),
            dim=2
        )

        output, hidden = self.gru(
            rnn_input,
            hidden
        )

        output = output.squeeze(1)

        context = context.squeeze(1)

        embedded = embedded.squeeze(1)

        prediction = self.fc_out(
            torch.cat(
                (
                    output,
                    context,
                    embedded
                ),
                dim=1
            )
        )

        return (
            prediction,
            hidden,
            attention_weights
        )


# ============================================================
# 14. SEQ2SEQ MODEL
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

        batch_size = source.size(0)

        target_length = target.size(1)

        target_vocab_size = (
            self.decoder.fc_out.out_features
        )

        outputs = torch.zeros(
            batch_size,
            target_length,
            target_vocab_size,
            device=self.device
        )

        encoder_outputs, hidden = (
            self.encoder(source)
        )

        source_mask = (
            source !=
            en_word_to_idx[PAD_TOKEN]
        )

        input_token = target[:, 0]

        for t in range(
            1,
            target_length
        ):

            output, hidden, _ = (
                self.decoder(
                    input_token,
                    hidden,
                    encoder_outputs,
                    source_mask
                )
            )

            outputs[:, t, :] = output

            best_guess = output.argmax(1)

            teacher_force = (
                random.random()
                < teacher_forcing_ratio
            )

            if teacher_force:

                input_token = target[:, t]

            else:

                input_token = best_guess

        return outputs


# ============================================================
# 15. CREATE MODEL
# ============================================================

INPUT_DIM = len(en_word_to_idx)

OUTPUT_DIM = len(fr_word_to_idx)

ENC_EMB_DIM = 128

DEC_EMB_DIM = 128

HID_DIM = 256


encoder = Encoder(
    INPUT_DIM,
    ENC_EMB_DIM,
    HID_DIM
)

decoder = AttentionDecoder(
    OUTPUT_DIM,
    DEC_EMB_DIM,
    HID_DIM
)

model = Seq2Seq(
    encoder,
    decoder,
    device
).to(device)


# ============================================================
# 16. OPTIMIZER AND LOSS
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.CrossEntropyLoss(
    ignore_index=fr_word_to_idx[PAD_TOKEN]
)


# ============================================================
# 17. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    epoch_loss = 0

    for source, target in loader:

        source = source.to(
            device,
            non_blocking=True
        )

        target = target.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        output = model(
            source,
            target,
            teacher_forcing_ratio=0.5
        )

        output_dim = output.size(-1)

        output = output[:, 1:, :].reshape(
            -1,
            output_dim
        )

        target = target[:, 1:].reshape(
            -1
        )

        loss = criterion(
            output,
            target
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1
        )

        optimizer.step()

        epoch_loss += loss.item()

    return (
        epoch_loss /
        len(loader)
    )


# ============================================================
# 18. VALIDATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for source, target in loader:

            source = source.to(
                device,
                non_blocking=True
            )

            target = target.to(
                device,
                non_blocking=True
            )

            output = model(
                source,
                target,
                teacher_forcing_ratio=0
            )

            output_dim = output.size(-1)

            output = output[:, 1:, :].reshape(
                -1,
                output_dim
            )

            target = target[:, 1:].reshape(
                -1
            )

            loss = criterion(
                output,
                target
            )

            epoch_loss += loss.item()

    return (
        epoch_loss /
        len(loader)
    )


# ============================================================
# 19. LOAD TRAINED ATTENTION MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        "best_attention_model.pt",
        map_location=device
    )
)

model.eval()

print("Best attention model loaded.")

# ============================================================
# 20. GREEDY TRANSLATION
# ============================================================

def translate_sentence(
    sentence,
    max_length=50
):

    model.eval()

    source_ids = sentence_to_ids(
        sentence,
        en_word_to_idx
    )

    source_tensor = torch.tensor(
        source_ids,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        encoder_outputs, hidden = (
            model.encoder(source_tensor)
        )

    source_mask = (
        source_tensor
        != en_word_to_idx[PAD_TOKEN]
    )

    input_token = torch.tensor(
        [fr_word_to_idx[SOS_TOKEN]],
        dtype=torch.long
    ).to(device)

    generated_tokens = []

    with torch.no_grad():

        for _ in range(max_length):

            output, hidden, attention_weights = (
                model.decoder(
                    input_token,
                    hidden,
                    encoder_outputs,
                    source_mask
                )
            )

            predicted_token = output.argmax(
                1
            ).item()

            if predicted_token == fr_word_to_idx[EOS_TOKEN]:
                break

            predicted_word = fr_idx_to_word.get(
                predicted_token,
                UNK_TOKEN
            )

            generated_tokens.append(
                predicted_word
            )

            input_token = torch.tensor(
                [predicted_token],
                dtype=torch.long
            ).to(device)

    return generated_tokens
def beam_search_translate(sentence, beam_width=3, max_length=50):

    model.eval()

    source_ids = sentence_to_ids(sentence, en_word_to_idx)

    source_tensor = torch.tensor(
        source_ids,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        encoder_outputs, hidden = model.encoder(source_tensor)

    source_mask = (
        source_tensor != en_word_to_idx[PAD_TOKEN]
    )

    sos_token = fr_word_to_idx[SOS_TOKEN]
    eos_token = fr_word_to_idx[EOS_TOKEN]

    # Each beam:
    # (generated_tokens, score, hidden_state)

    beams = [
        ([sos_token], 0.0, hidden)
    ]

    for _ in range(max_length):

        candidates = []

        for tokens, score, beam_hidden in beams:

            # If this beam already ended, keep it
            if tokens[-1] == eos_token:
                candidates.append(
                    (tokens, score, beam_hidden)
                )
                continue

            input_token = torch.tensor(
                [tokens[-1]],
                dtype=torch.long
            ).to(device)

            with torch.inference_mode():

                output, new_hidden, _ = model.decoder(
                    input_token,
                    beam_hidden,
                    encoder_outputs,
                    source_mask
                )

                log_probs = torch.log_softmax(
                    output,
                    dim=1
                )

                top_log_probs, top_indices = torch.topk(
                    log_probs,
                    beam_width,
                    dim=1
                )

            for j in range(beam_width):

                next_token = top_indices[0][j].item()
                next_score = (
                    score +
                    top_log_probs[0][j].item()
                )

                candidates.append(
                    (
                        tokens + [next_token],
                        next_score,
                        new_hidden.clone()
                    )
                )

        # Keep best beam_width candidates
        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        beams = candidates[:beam_width]

        # Stop if all beams finished
        if all(
            beam[0][-1] == eos_token
            for beam in beams
        ):
            break

    # Best final beam
    best_beam = max(
        beams,
        key=lambda x: x[1]
    )

    best_tokens = best_beam[0]

    generated_tokens = []

    for token_id in best_tokens:

        if token_id == sos_token:
            continue

        if token_id == eos_token:
            break

        word = fr_idx_to_word.get(
            token_id,
            UNK_TOKEN
        )

        generated_tokens.append(word)

    return generated_tokens

# ============================================================
# 21. SAMPLE TRANSLATIONS
# ============================================================

print()
print("Sample translations")
print("=" * 60)

for i in range(5):

    source_sentence = test_df.iloc[i]["en"]
    target_sentence = test_df.iloc[i]["fr"]

    prediction = translate_sentence(
        source_sentence
    )

    print()
    print("SOURCE:")
    print(source_sentence)

    print()
    print("REFERENCE:")
    print(target_sentence)

    print()
    print("PREDICTION:")
    print(" ".join(prediction))

    print("-" * 60)

# ============================================================
# 22. BLEU EVALUATION
# ============================================================

references = []
hypotheses = []

print()
print("Calculating BLEU score...")
print("Test samples:", len(test_df))

for i in range(len(test_df)):

    source_sentence = test_df.iloc[i]["en"]
    target_sentence = test_df.iloc[i]["fr"]

    prediction = translate_sentence(
        source_sentence
    )

    reference_tokens = tokenize(
        target_sentence
    )

    references.append(
        [reference_tokens]
    )

    hypotheses.append(
        prediction
    )

    if (i + 1) % 500 == 0:
        print(
            f"Processed {i + 1}/{len(test_df)}"
        )


bleu_score = corpus_bleu(
    references,
    hypotheses
)

print()
print("=" * 60)
print("TASK 2 ATTENTION RESULTS")
print("=" * 60)

print(
    f"Attention Seq2Seq BLEU: {bleu_score:.4f}"
)

print(
    "Task 1 Basic Seq2Seq BLEU: 0.0159"
)

print("=" * 60)

sample_sentences = [
    "Several years ago here at TED, Peter Skillman introduced a design challenge called the marshmallow challenge.",
    "And the idea's pretty simple: Teams of four have to build the tallest free-standing structure out of 20 sticks of spaghetti, one yard of tape, one yard of string and a marshmallow.",
    "The marshmallow has to be on top.",
    "And, though it seems really simple, it's actually pretty hard because it forces people to collaborate very quickly.",
    "And so, I thought this was an interesting idea, and I incorporated it into a design workshop."
]

for sentence in sample_sentences:

    prediction = beam_search_translate(
        sentence,
        beam_width=3
    )

    print("\nSOURCE:")
    print(sentence)

    print("\nBEAM SEARCH:")
    print(" ".join(prediction))

    print("-" * 60)