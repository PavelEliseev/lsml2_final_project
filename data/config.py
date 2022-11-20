import transformers
from transformers import BertTokenizer

DEVICE = 'cpu'
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
MODEL_PATH = "model.bin"
TRAINING_FILE = "IMDB Dataset.csv"
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
