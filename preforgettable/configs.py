DEFAULT_MODEL_NAME = "google/vit-base-patch16-224"
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
INFERENCE_MODES = (
    "full_bank_shuffle",
    "match_only",
    "mixed_subset",
    "all_wrong",
    "wrong_subset",
    "random_wrong",
    "none",
    "infer_best",
)
