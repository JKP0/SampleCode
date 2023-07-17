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

import os
import csv
import urllib.request

from transformers import AutoTokenizer, AutoConfig

from transformers import RobertaTokenizer
from transformers import RobertaConfig, RobertaModel

import numpy as np
from scipy.special import softmax


def preprocess_tweet(text):
    """Preprocess text (username and link placeholders)"""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def process_sample_io(tokenizer, model, labels_data, text):
    text = "Covid cases are increasing fast!" if text is None else text

    encoded_input = tokenizer(preprocess_tweet(text), return_tensors='tf')
    # print(encoded_input, "already shaped for list input")
    output = model(encoded_input)
    scores = output[0][0].numpy()
    scores = softmax(scores)

    # Print labels and scores
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        k = labels_data[ranking[i]]
        s = scores[ranking[i]]
        print(f"{i + 1}) {k} {np.round(float(s), 4)}")

    return True


def get_default_model_and_tokenizer(use_auto_tokenizer=False):
    #  model/tokenizer names: "roberta-base", "distilbert-base-uncased" or "stevhliu/my_awesome_model"
    tokenizer_model_name = "roberta-base"

    # texts = [" Hello world" "Hello world"]

    # Initializing a RoBERTa tokenizer
    if use_auto_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        # encodings = tokenizer(texts, return_tensors="np")

    else:
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_model_name)

    # Initializing a RoBERTa configuration
    configuration = RobertaConfig(tokenizer_model_name)

    # Initializing a model (with random weights) from the configuration
    model = RobertaModel(configuration)

    # Accessing the model configuration
    configuration = model.config

    # tokenizer(texts[0])["input_ids"]  # [0, 31414, 232, 2]
    # tokenizer(texts[1])["input_ids"] # [0, 20920, 232, 2]

    return tokenizer, model, configuration


def _get_pt_model(model_name, model_save_path):
    raise NotImplementedError("Function Not Implemented.")

    return config, model


def _get_tf_model(model_name, model_save_path):
    raise NotImplementedError("Function Not Implemented.")

    return config, model


def get_pretrained_model_and_tokenizer(model_name, pre_train_model_dir="./",
                                       sample_text=None, do_process_sample_io=True):
    """
    :param do_process_sample_io:
        True/False: Decides that model should print outputs on sample text or not
    :param sample_text:
        None or Some text to process sample i-o. if None default text will be utilized.
    :param pre_train_model_dir:
        str/byt: Directory to save pre-trained model
    :param model_name:
        PreTrained model name, which will be used for model checkpoint download.
    :return: tokenizer, model, config
    """
    #  #############################################################################################
    # Not Used
    # model_name = "distilbert-base-uncased"
    # model = TFAutoModelForSequenceClassification.from_pretrained(model_name,
    #                                                             num_labels=2, id2label=id2label,
    #                                                             label2id=label2id)
    #  #############################################################################################

    if model_name is None:
        return get_default_model_and_tokenizer()

    tokenizer_save_path = os.path.join(pre_train_model_dir, "tokenizer", model_name)
    model_save_path = os.path.join(pre_train_model_dir, "model", model_name)

    def download_tokenizer():
        print("Downloading [{}] Tokenizer From Source.".format(model_name))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(tokenizer_save_path)

        return tokenizer

    if os.path.exists(tokenizer_save_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
            print("Tokenizer Loaded From Local Directory.\n [{}]".format(tokenizer_save_path))

        except Exception as exp:
            print("Failed To Load Tokenizer From Local Directory.\n {}\nError:\n{}".format(tokenizer_save_path, exp))
            os.remove(tokenizer_save_path)
            tokenizer = download_tokenizer()
    else:
        tokenizer = download_tokenizer()

    #  ################################ PT Block ###########################################
    #  config, model = _get_pt_model(model_name, model_save_path)
    #  ################################ PT Block ###########################################

    #  ################################ TF Block ###########################################
    config, model = _get_tf_model(model_name, model_save_path)

    #  ################################ TF Block ###########################################

    if model_name == "cardiffnlp/twitter-roberta-base-sentiment":
        # download label mapping
        task = "sentiment"
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csv_reader = csv.reader(html, delimiter='\t')

        labels = [row[1] for row in csv_reader if len(row) > 1]

        config.id2label2 = labels

    else:
        labels = config.id2label
        config.id2label2 = []

    if do_process_sample_io:
        process_sample_io(tokenizer, model, labels, sample_text)

    return tokenizer, config, model


if __name__ == "__main__":
    text_to_test_tokenizer = "I know to Good Bye, but not about COVID"
    print_text = ["TokenizerCall", "ModelSummary", "ModelConfig"]
    task_sub = "sentiment"
    model_nm = f"xyz1_{task_sub}_abc1"
    print("[1] Trying for model: [{}]\n".format(model_nm))
    tokenizer_is, model_is, config_is = get_pretrained_model_and_tokenizer(model_nm)
    print(f"{print_text[0]}\n",
          tokenizer_is(text_to_test_tokenizer, return_tensors="np"), "\n",
          tokenizer_is(print_text, return_tensors="np"))
    print(f"\n{print_text[1]}\n", model_is.summary())
    print(f"\n{print_text[2]}\n", config_is)

    model_nm = f"xyz2_{task_sub}_abc2"
    print("[2] Trying for model: [{}]\n".format(model_nm))
    tokenizer_is, model_is, config_is = get_pretrained_model_and_tokenizer(model_nm)
    print(f"{print_text[0]}\n", tokenizer_is(text_to_test_tokenizer, return_tensors="np"))
    print(f"\n{print_text[1]}\n", model_is.summary())
    print(f"\n{print_text[2]}\n", config_is)

    model_nm = f"xyz3_{task_sub}_abc3"
    print("[3] Trying for model: [{}]\n".format(model_nm))
    tokenizer_is, model_is, config_is = get_pretrained_model_and_tokenizer(model_nm)
    print(f"{print_text[0]}\n", tokenizer_is(text_to_test_tokenizer, return_tensors="np"))
    print(f"\n{print_text[1]}\n", model_is.summary())
    print(f"\n{print_text[2]}\n", config_is)
