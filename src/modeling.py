
import numpy as np

import tensorflow as tf

from pre_trained_model_opps import get_pretrained_model_and_tokenizer, preprocess_tweet

from utils import MlU


class C3Model(tf.keras.Model):
    def get_tokenized_features(self, texts):
        texts = [preprocess_tweet(text) for text in texts]

        tokens = self.tokenizer(texts, return_tensors=self.tokenizer_tensor_type)

        return tokens

    def call(self, texts, training=None, mask=None):
        raise NotImplementedError("Model Call Not Implemented")

        return features

    def __init__(self, model_name=None, pre_classifier_fc_units=32, pre_classifier_activation=None,
                 number_of_classes=13, classifier_activation=None):
        # super(C3Model, self).__init__()
        self.tokenizer_tensor_type = "tf"
        self.tune_pretraining = False

        if model_name is None:
            model_name = MlU.model_name_roberta_sentiment_latest

        tokenizer, config, model = get_pretrained_model_and_tokenizer(model_name, MlU.pre_train_model_dir)

        self.tokenizer = tokenizer
        self.pre_training_config = config
        self.pre_trained_model = model

        self.pre_classifier = tf.keras.layers.Dense(pre_classifier_fc_units, activation=pre_classifier_activation)
        self.feature_flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.layers.Dense(number_of_classes, activation=classifier_activation)


class C3Processor(object):
    def load_intermediate_weights(self, weight_path):
        self.model.load_weights(weight_path)
        return True

    def save_model(self, model_save_dir_info, save_h5=False):
        self.model.save_weights(MlU.make_model_ckpt_path(*model_save_dir_info))

        if save_h5:
            self.model.save(MlU.make_model_h5_save_path(model_save_dir_info[0]))

        return True

    @classmethod
    def calculate_accuracy(cls, actual_y, predicted_y):
        intent_acc = tf.reduce_mean(tf.cast(tf.equal(actual_y, predicted_y), tf.float32))

        return intent_acc

    def calculate_loss(self, actual_y, predicted_y):
        loss = self.loss_class_instant(actual_y, predicted_y)

        return loss

    def train_step(self, texts, labels, training, mask):
        with tf.GradientTape() as tape:
            y_predicted = self.model(texts, training=training, mask=mask)

            loss = self.calculate_loss(labels, y_predicted)
            acc = self.calculate_accuracy(labels, y_predicted)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, acc

    def __call__(self, texts, labels, training=None, mask=None):
        try:
            assert len(texts) == len(labels)
        except AssertionError as exp:
            print("Text & Label count/batch-size should be same.")
            raise exp

        labels = np.array(labels)

        if training:
            return self.train_step(texts, labels, training, mask)

        y_predicted = self.model(texts, training=training, mask=mask)

        loss = self.calculate_loss(labels, y_predicted)
        acc = self.calculate_accuracy(labels, y_predicted)

        return loss, acc

    def __init__(self, model_name, pre_classifier_fc_units=32, pre_classifier_activation=None,
                 number_of_classes=13, classifier_activation=None, learning_rate=0.0001, loss_name="scce"):

        self.model_name = model_name
        self.model = C3Model(model_name=model_name,
                             pre_classifier_fc_units=pre_classifier_fc_units,
                             pre_classifier_activation=pre_classifier_activation,
                             number_of_classes=number_of_classes,
                             classifier_activation=classifier_activation)

        print(self.model.summary())

        if loss_name == "cce":
            self.loss_class_instant = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, name='categorical_crossentropy')
        else:
            self.loss_class_instant = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, name='sparse_categorical_crossentropy')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


class C3InferenceInterface(object):
    def do_prediction(self, texts, labels=None):
        y_predict = np.argmax(self.model(texts).numpy(), axis=-1)

        predict_labels = [self.id_to_labels[e] for e in y_predict]

        if labels is not None:
            labels = np.array([self.labels_to_id[label] for label in labels])
            acc = C3Processor.calculate_accuracy(labels, y_predict)
        else:
            acc = -1.0

        return predict_labels, acc

    def __init__(self, labels_to_read, fine_tuned_ckpt_path, model_name,
                 pre_classifier_fc_units=32, pre_classifier_activation=None,
                 number_of_classes=13, classifier_activation=None):

        self.model_name = model_name

        if ".h5" in fine_tuned_ckpt_path:
            self.model = tf.keras.saved_model.load(fine_tuned_ckpt_path)

        else:
            self.model = C3Model(model_name=model_name,
                                 pre_classifier_fc_units=pre_classifier_fc_units,
                                 pre_classifier_activation=pre_classifier_activation,
                                 number_of_classes=number_of_classes,
                                 classifier_activation=classifier_activation)

            self.model.load_weights(fine_tuned_ckpt_path)

            self.model.save(MlU.make_model_h5_save_path(model_name))

        self.id_to_labels = labels_to_read
        self.labels_to_id = {v: e for e, v in labels_to_read.items()}
