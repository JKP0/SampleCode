import numpy as np

"""
Models To Experiments
1. Zero-Shot Classification
    1. [BART] facebook/bart-large-mnli
    2. [DeBERTa] sileod/deberta-v3-base-tasksource-nli
    3. [XLM] joeddav/xlm-roberta-large-xnli
    4. [DeBERTa] MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
2. Text Classification
    1. [DistilBERT] distilbert-base-uncased-finetuned-sst-2-english
    2. [RoBERTa] cardiffnlp/twitter-roberta-base-sentiment
    3. [DistilRoBERTa] j-hartmann/emotion-english-distilroberta-base
    4. [RoBERTa] cardiffnlp/twitter-roberta-base-sentiment-latest
    5. [BERT] nlptown/bert-base-multilingual-uncased-sentiment
    6. [XLM-RoBERTa] cardiffnlp/twitter-xlm-roberta-base-sentiment
    7. [XLM-RoBERTa] papluca/xlm-roberta-base-language-detection
    8. [FinBERT] ProsusAI/finbert 
"""


class C3Model():
    pass


class C3Processor(object):
    def calculate_loss(self, actual_y, predicted_y):
        pass

    def calculate_accuracy(self, actual_y, predicted_y):
        pass

    def __call__(self, features, training=None, mask=None):
        return self.model(features, training=training, mask=mask)

    def __init__(self, model):
        self.model = model


class C3ProcessorE2E(C3Processor):
    def do_predict(self, texts, labels=None):
        features = self.tokenizer(texts)

        y_predict = np.argmax(self.model(features).numpy(), axis=-1)

        if labels is None:
            return [self.labels_to_id[e] for e in y_predict]

        predict_labels = [self.labels_to_id[e] for e in y_predict]
        labels = np.array([self.labels_to_id[label] for label in labels])

        acc = self.calculate_accuracy(labels, predict_labels)

        return predict_labels, acc

    def __init__(self, model, tokenizer, labels_to_read):
        super(C3ProcessorE2E, self).__init__(model=model)

        self.tokenizer = tokenizer
        self.id_to_labels = labels_to_read
        self.labels_to_id = {v: e for e, v in labels_to_read.items()}
