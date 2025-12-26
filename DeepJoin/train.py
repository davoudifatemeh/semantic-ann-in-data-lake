import torch, random, numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from pairsBuilder import PairsBuilder, split_pairs, save_split
from config import SEED, POS_PAIRS_OUTPUT, ANNOTATED, SPLIT_OUTPUT_PREFIX, MODEL_OUTPUT_PATH, TRAIN_RATIO, MODEL_NAME, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, WARMUP_STEPS, PATIENCE, SHUFFLE_RATE, HEADER_INFO_WITH_SYNS

# Reproducibility
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def make_examples(pairs, shuffle=False):
    # Create InputExamples for MNR training using only positives.
    skiped = 0
    examples = []
    for a, b in pairs:
        if a is None or b is None:
            skiped += 1
            continue
        
        examples.append(InputExample(texts=[a, b]))
    if skiped > 0:
        print(f"Skipped {skiped} pairs with None values.")
    return examples

def make_eval_pairs(pairs):
    # Prepare evaluator data with both positive and negative pairs.
    s1, s2, y = [], [], []
    for p in pairs:
        if p[2] in (0, 1):  # ensure label is valid
            s1.append(str(p[0]))
            s2.append(str(p[1]))
            y.append(float(p[2]))
    return s1, s2, y

# 3. Data preparation
builder = PairsBuilder(ANNOTATED, use_annotation=True, include_header=True, corrupt_header=False)
positive_pairs = builder.build_pairs()
# PairsBuilder.save_all_pairs(positive_pairs, out_file=POS_PAIRS_OUTPUT, fmt="jsonl")
train_pairs, test_pairs = split_pairs(positive_pairs, train_ratio=TRAIN_RATIO, seed=SEED)
save_split(test_pairs, out_prefix=SPLIT_OUTPUT_PREFIX, fmt="jsonl")
train_examples = make_examples(train_pairs, shuffle=False) # only positives, with shuffling. 
train_loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

# 4. Model setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = SentenceTransformer(MODEL_NAME, device=device)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# 5. Training
optimizer_params = {'lr': LEARNING_RATE, 'eps': 1e-6, 'weight_decay': WEIGHT_DECAY}

model.fit(
    train_objectives=[(train_loader, train_loss)],
    epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS,
    optimizer_params=optimizer_params,
    show_progress_bar=True,
    output_path=None,  # defer saving
    use_amp=True
)

# 6. Save the fine-tuned model
model.save(MODEL_OUTPUT_PATH)
print(f"Model saved at {MODEL_OUTPUT_PATH}.")

