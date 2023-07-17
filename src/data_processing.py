import random

from utils import DrU


class DataProcessor(object):
    def enable_training_mode(self, data_df, validation_ids, batch_size):
        assert all([e in data_df.columns for e in (DrU.data_col_tweet, DrU.add_col_train_label)])

        self.data_df = data_df
        self.batch_size = batch_size

        self.validation_ids = list(set(validation_ids))
        self.training_ids = [i for i in range(data_df.shape[0]) if i not in validation_ids]

        assert sum([len(self.validation_ids), len(self.training_ids)]) == data_df.shape[0]

        batch_count = len(self.training_ids) / batch_size
        if batch_count > int(batch_count):
            self.training_batch_count = int(batch_count + 1)

        else:
            self.training_batch_count = int(batch_count)

        batch_count = len(self.validation_ids) / batch_size
        if batch_count > int(batch_count):
            self.validation_batch_count = int(batch_count + 1)

        else:
            self.validation_batch_count = int(batch_count)

        return True

    def enable_test_model(self, data_df, batch_size):
        self.test_data_df = data_df
        self.batch_size = batch_size

        test_batch_count = data_df.shape[0]/self.batch_size
        if test_batch_count > int(test_batch_count):
            test_batch_count += 1

        self.test_batch_count = test_batch_count

        return True

    def on_epoch_end(self, type_id=-1):
        if type_id in (-1, 0):
            random.shuffle(self.training_ids)

        if type_id in (-1, 1):
            random.shuffle(self.validation_ids)

        return True

    def _get_data_for_index_ids(self, ids_to_read):
        texts = [self.data_df.at[i, DrU.data_col_tweet] for i in ids_to_read]
        labels = [self.data_df.at[i, DrU.add_col_train_label] for i in ids_to_read]

        return texts, labels

    def get_training_items(self, batch_index):
        start = self.batch_size * batch_index
        end = min([start + self.batch_size, len(self.training_ids)])

        return self._get_data_for_index_ids(self.training_ids[start: end])

    def get_validation_items(self, batch_index):
        start = self.batch_size * batch_index
        end = min([start + self.batch_size, len(self.validation_ids)])

        return self._get_data_for_index_ids(self.validation_ids[start: end])

    def get_test_items(self, batch_index):
        start = self.batch_size * batch_index
        end = min([start + self.batch_size, len(self.test_data_df.shape[0])])
        ids_to_read = list(range(start, end))

        texts = [self.data_df.at[i, DrU.data_col_tweet] for i in ids_to_read]
        labels = [self.data_df.at[i, DrU.add_col_train_label] for i in ids_to_read]

        return texts, labels

    def get_training_data(self):
        if self.training_batch_count == 1:
            raise AttributeError("Training Model Not Enabled.")

        for batch_id in range(self.training_batch_count):
            texts, labels = self.get_training_items(batch_id)
            yield batch_id, texts, labels

        self.on_epoch_end(0)

    def get_validation_data(self):
        for batch_id in range(self.validation_batch_count):
            texts, labels = self.get_validation_items(batch_id)
            yield batch_id, texts, labels

        self.on_epoch_end(1)

    def get_test_data(self):
        for batch_id in range(self.test_batch_count):
            texts, labels = self.get_test_items(batch_id)
            yield batch_id, texts, labels

    def __init__(self):
        self.data_df = None
        self.test_data_df = None

        self.training_ids = None
        self.validation_ids = None
        self.batch_size = 1
        self.training_batch_count = 1
        self.validation_batch_count = 1
        self.test_batch_count = 1
