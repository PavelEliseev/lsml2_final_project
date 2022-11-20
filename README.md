# LSML FINAL PROJECT - comment sentiment analysis

# Introduction
This is an ML service to return sentiment based on the comment. Project consists of Transformers (for fine-tuning BERT model) and Pytorch. Flask was used to simple web server where user can submit simple json with comment as an input and get result for sentiment analysis. Docker was used to containerize the application.

# Dataset

https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

IMDB dataset having 50K movie reviews for natural language processing or Text analytics.
This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms.

# Fine-tuning BERT

I used BERT model for fine-tuning: https://huggingface.co/bert-base-uncased

I take optimal parametres for fine tuning:
- epochs : 2
- batch size: 8, 4
- lr: 3e-5
- optimizer: Adamw

I used different batch size, because of collab gpu memmory limit, but on this parametres we already have 0.93 accuracy
link: https://arxiv.org/pdf/1810.04805.pdf (A.3)

The data folder includes all reqired .py files and .ipynb file with fine-tuning proccedure (stored as model.bin, so you don't need to train it again)

# Installation
After cloning this repo to yours machine, from this repo:

```
docker-compose build
```
 
```
docker-compose up -d
```

# Testing API

After Installation you can submit .json files to http://localhost:5000/predict to test the analysis

For Windows:
```
curl -H "Content-Type: application/json" -X POST http://localhost:5000/predict -d "{\"text\":\"YOUR TEXT\"}"
```

For Unix:

```
curl -X POST -H "Content-Type: application/json" -d '{"text": "YOUR TEXT"}' http://localhost:5000/predict
```

As result you will see:
```
{"Prediction":"Positive"} 
```

```
{"Prediction":"Negative"} 
```

It depends of text you submited.

# Result points:
From list fo cretaria we have:
- 1. Design document and dataset description - 1 point max
- 2. Model training code - 2 points max
  - 2.1. Jupyter Notebook - 1 point
- 3. Dockerfile - 6 points max
  - 3.1. synchronous projects - 1 point
  - 3.2. REST API / Telegram Bot - 1 point
   - 3.3. model transfer learning - 1 point

I made these points
