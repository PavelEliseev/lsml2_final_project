import config
from model import BERTBaseUncased
import torch
import os
from flask import Flask, request, jsonify


app = Flask(__name__)

MODEL = None
DEVICE = config.DEVICE
PREDICTION_DICT = dict()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    sentence = list(data.values())
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())
    inputs = tokenizer.encode_plus(review,
                                   None,
                                   add_special_tokens=True,
                                   max_length=max_len
                                   )
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)
    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)
    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    pos_prediction = outputs[0][0]
    neg_prediction = 1 - pos_prediction
    result = 'Negative' if pos_prediction < neg_prediction else 'Positive'
    output = {'Prediction': result}
    return jsonify(output)


if __name__ == '__main__':
    MODEL = BERTBaseUncased()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    MODEL.to(DEVICE)
    MODEL.eval()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
