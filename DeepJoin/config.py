########## Configurations ##########
APP_NAME = "rodi"
ANNOTATED = f"data/{APP_NAME}/{APP_NAME}_join_ground_truth.csv"
ANNOTATED_TEST = f"data/{APP_NAME}/{APP_NAME}_join_ground_truth_test.csv"
POS_PAIRS_OUTPUT = f"data/{APP_NAME}/all_pos_pairs.jsonl"
SCHEMA = f"paths/to/{APP_NAME}_csv_schema.json"
HEADER_INFO_WITH_SYNS = "paths/to/header_info_with_synonyms.json"
ANNOTATE_CORRUPTION = 0  # 0: no corruption, 1: type 1, 2: type 2, 3: type 3
CORRUPTION_WITH_ANNOTATIONS = "paths/to/file/for/corrupted_annotations.json"
ANNOTATE_WITHOUT_HEADER = False
NO_HEADER_ANNOTATIONS = "paths/to/file/for/no_header_annotations.json"
SEED = 42
TRAIN_RATIO=0.8
SPLIT_OUTPUT_PREFIX = ""
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

#---------- training parameters ----------
MODEL_OUTPUT_PATH   = "paths/to/save/finetuned_model"
SHUFFLE_RATE  = 0.2
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 10000
PATIENCE = 2 

# #---------- teting parameters ----------
TEST_REPO = "path/to/file/for/positive_test_pairs.jsonl"
QUERY_FILE = "path/to/file/for/queries.jsonl"
INDEX_PATH = "column_index"  
TOP_K = 5