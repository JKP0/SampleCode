
import random
import numpy as np

from constants import DrC


class DataProcessor(object):
    def set_to_training_mode(self, data_df, validation_ids, batch_size):
        assert all([e in data_df.columns for e in (DrC.data_col_tweet, DrC.add_col_train_label)])

        self.data_df = data_df
        self.batch_size = batch_size

        self.validation_ids = list(set(validation_ids))
        self.training_ids = [i for i in range(data_df.shape[0]) if i not in validation_ids]

        assert sum([len(self.validation_ids), len(self.training_ids)]) == data_df.shape[0]

        batch_count = len(self.training_ids)/batch_size
        if batch_count > int(batch_count):
            self.training_batch_count = int(batch_count+1)

        else:
            self.training_batch_count = int(batch_count)

        batch_count = len(self.validation_ids)/batch_size
        if batch_count > int(batch_count):
            self.validation_batch_count = int(batch_count+1)

        else:
            self.validation_batch_count = int(batch_count)

        self.is_training_mode_set = True

        return self.is_training_mode_set

    def on_epoch_end(self, type_id=-1):
        if type_id in (-1, 0):
            random.shuffle(self.training_ids)

        if type_id in (-1, 1):
            random.shuffle(self.validation_ids)

        return True

    def encode_tweet(self, text, label=None):
        token = self.tokenizer(text)

        if label is None:
            return token, np.array([-1])

        label = np.array([self.labels_to_id[label]])

        return token, label

    def encode_tweets(self, texts, labels=None):
        token = self.tokenizer(texts)

        if labels is None:
            return token, np.array([-1 for _ in texts])

        labels = np.array([self.labels_to_id[label] for label in labels])

        return token, labels

    def encode_tweets_and_labels(self, texts, labels):
        token = self.tokenizer(texts)

        labels = np.array([self.labels_to_id[label] for label in labels])

        return token, labels

    def _get_data_for_index_ids(self, ids_to_read):
        texts = [self.data_df.at[i, DrC.data_col_tweet] for i in ids_to_read]
        labels = [self.data_df.at[i, DrC.add_col_train_label] for i in ids_to_read]

        return self.encode_tweets_and_labels(texts, labels)

    def get_training_items(self, batch_index):
        start = self.batch_size * batch_index
        end = min([start + self.batch_size, len(self.training_ids)])

        return self._get_data_for_index_ids(self.training_ids[start: end])

    def get_validation_items(self, batch_index):
        start = self.batch_size * batch_index
        end = min([start + self.batch_size, len(self.training_ids)])

        return self._get_data_for_index_ids(self.validation_ids[start: end])

    def get_training_data(self):
        for batch_id in range(self.training_batch_count):
            yield self.get_training_items(batch_id)

        self.on_epoch_end(0)

    def get_validation_data(self):
        for batch_id in range(self.validation_batch_count):
            yield self.get_training_items(batch_id)

        self.on_epoch_end(1)

    def __init__(self, tokenizer=None, label_to_read=None):
        if tokenizer is None:
            tokenizer = "ABC assign tokenizer class"

        self.tokenizer = tokenizer

        self.id_to_labels = label_to_read
        self.labels_to_id = {v: e for e, v in label_to_read.items()}

        assert len(self.labels_to_id) == len(self.labels_to_id)

        self.is_training_mode_set = False
        self.data_df = None
        self.training_ids = None
        self.validation_ids = None
        self.batch_size = 1
        self.training_batch_count = 1
        self.validation_batch_count = 1
