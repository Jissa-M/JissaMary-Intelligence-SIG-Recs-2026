import pandas as pd
import re
import torch
import torch.nn as nn

from collections import Counter
from nltk.translate.bleu_score import corpus_bleu


# ============================================================
# 1. LOAD TEST DATA
# ============================================================

test_df = pd.read_csv("test.csv")

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

        for token in tokenize(sentence):
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


# IMPORTANT:
# These must be built exactly the same way as during training.

train_df = pd.read_csv("train.csv")

en_word_to_idx, en_idx_to_word = build_vocab(
    train_df["en"]
)

fr_word_to_idx, fr_idx_to_word = build_vocab(
    train_df["fr"]
)

print("English vocabulary:", len(en_word_to_idx))
print("French vocabulary:", len(fr_word_to_idx))


# ============================================================
# 5. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# 6. ENCODER
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
# 7. DECODER
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
# 8. CREATE MODEL
# ============================================================

INPUT_DIM = len(en_word_to_idx)
OUTPUT_DIM = len(fr_word_to_idx)

ENC_EMB_DIM = 128
HIDDEN_DIM = 256


encoder = Encoder(
    INPUT_DIM,
    ENC_EMB_DIM,
    HIDDEN_DIM
).to(device)

decoder = Decoder(
    OUTPUT_DIM,
    ENC_EMB_DIM,
    HIDDEN_DIM
).to(device)


# ============================================================
# 9. LOAD TRAINED MODEL
# ============================================================

encoder.load_state_dict(
    {
        key.replace("encoder.", "", 1): value
        for key, value in torch.load(
            "best_seq2seq_model.pt",
            map_location=device
        ).items()
        if key.startswith("encoder.")
    }
)

decoder.load_state_dict(
    {
        key.replace("decoder.", "", 1): value
        for key, value in torch.load(
            "best_seq2seq_model.pt",
            map_location=device
        ).items()
        if key.startswith("decoder.")
    }
)

encoder.eval()
decoder.eval()

print("Best model loaded successfully.")


# ============================================================
# 10. TRANSLATION FUNCTION
# ============================================================

def translate_sentence(
    sentence,
    max_length=50
):

    tokens = tokenize(sentence)

    tokens = (
        [SOS_TOKEN]
        + tokens
        + [EOS_TOKEN]
    )

    source_ids = []

    for token in tokens:

        if token in en_word_to_idx:
            source_ids.append(
                en_word_to_idx[token]
            )
        else:
            source_ids.append(
                en_word_to_idx[UNK_TOKEN]
            )

    source_tensor = torch.tensor(
        source_ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():

        hidden = encoder(
            source_tensor
        )

    input_token = torch.tensor(
        [fr_word_to_idx[SOS_TOKEN]],
        dtype=torch.long,
        device=device
    )

    generated_tokens = []

    for _ in range(max_length):

        with torch.no_grad():

            output, hidden = decoder(
                input_token,
                hidden
            )

        best_guess = output.argmax(
            dim=1
        ).item()

        predicted_token = fr_idx_to_word[
            best_guess
        ]

        if predicted_token == EOS_TOKEN:
            break

        if predicted_token != SOS_TOKEN:
            generated_tokens.append(
                predicted_token
            )

        input_token = torch.tensor(
            [best_guess],
            dtype=torch.long,
            device=device
        )

    return generated_tokens


# ============================================================
# 11. BLEU SCORE
# ============================================================

references = []
hypotheses = []

MAX_TEST_SAMPLES = 8597

for index in range(
    min(MAX_TEST_SAMPLES, len(test_df))
):

    source_sentence = test_df.iloc[index]["en"]
    target_sentence = test_df.iloc[index]["fr"]

    prediction = translate_sentence(
        source_sentence
    )

    reference = tokenize(
        target_sentence
    )

    references.append(
        [reference]
    )

    hypotheses.append(
        prediction
    )

    if index % 500 == 0:
        print(
            f"Evaluated {index}/{MAX_TEST_SAMPLES}"
        )


bleu_score = corpus_bleu(
    references,
    hypotheses
)


print()
print("==============================")
print("SUBTASK 1 RESULTS")
print("==============================")
print(
    f"BLEU Score: {bleu_score:.4f}"
)


# ============================================================
# 12. SAMPLE TRANSLATIONS
# ============================================================

print()
print("==============================")
print("SAMPLE TRANSLATIONS")
print("==============================")


for index in range(10):

    source_sentence = test_df.iloc[index]["en"]
    target_sentence = test_df.iloc[index]["fr"]

    prediction = translate_sentence(
        source_sentence
    )

    print()
    print("English :", source_sentence)
    print("Actual  :", target_sentence)
    print(
        "Predicted:",
        " ".join(prediction)
    )