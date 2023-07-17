import os
import argparse
import random

from src.modeling import C3Processor, C3InferenceInterface
from src.trainer import TrainingProcessor
from src.data_processing import DataProcessor
from src.data_read import DataInitialView

from src.utils import PsU


def split_dataset(total_sample_count, validation_fraction=0.18, random_seed=5787):
    completed_sample_index = list(range(total_sample_count))
    validation_sample = int(validation_fraction * total_sample_count)
    random.seed(random_seed)
    validation_index = random.sample(completed_sample_index, k=validation_sample)

    return validation_index


def train_model(model_name, model_config, epochs, batch_size, restart_training, verbose=True):
    validation_fraction = model_config[PsU.key_validation_fraction]
    training_monitor = model_config[PsU.key_training_monitor]

    data_reader = DataInitialView()
    validation_ids = split_dataset(data_reader.data_df.shape[0], validation_fraction)

    data_processor = DataProcessor()
    data_processor.enable_training_mode(data_df=data_reader.data_df,
                                        validation_ids=validation_ids,
                                        batch_size=batch_size)

    model = C3Processor(model_name=model_name,
                        pre_classifier_fc_units=model_config[PsU.key_pre_classifier_fc_units],
                        pre_classifier_activation=model_config[PsU.key_pre_classifier_activation],
                        number_of_classes=len(data_reader.labels_to_read),
                        classifier_activation=model_config[PsU.key_classifier_activation],
                        learning_rate=model_config[PsU.key_learning_rate],
                        loss_name=model_config[PsU.key_loss_name])

    trainer = TrainingProcessor(data_processor, model, epochs,
                                monitor_key=training_monitor, verbose=verbose)
    start_epoch = 0
    if restart_training:
        start_epoch = 1

    status = trainer.train(start_epoch)

    return status


def test_model(model_name, model_config, batch_size):
    data_reader = DataInitialView()

    data_processor = DataProcessor()
    data_processor.enable_test_model(data_reader.test_data_df, batch_size)

    pre_classifier_fc_units = model_config[PsU.key_pre_classifier_fc_units]
    pre_classifier_activation = model_config[PsU.key_pre_classifier_activation]
    number_of_classes = len(data_reader.labels_to_read)
    classifier_activation = model_config[PsU.key_classifier_activation]

    fine_tuned_ckpt_path = PsU.get_final_model_h5_path(model_name)
    if not os.path.exists(fine_tuned_ckpt_path):
        path = PsU.get_training_checkpoint_file(model_name)
        if not os.path.exists(path):
            raise ValueError("Model Training Not Found At-\nPath: [{}]".format(path))

        fine_tuned_ckpt_path, epoch = PsU.get_final_model_ckpt_path(model_name)
        print(f"H5  Model Save Not Found.\nModel Supposed To Get Loaded From {epoch} epoch Training Checkpoint.")

    model = C3InferenceInterface(data_reader.labels_to_read, fine_tuned_ckpt_path, model_name,
                                 pre_classifier_fc_units, pre_classifier_activation,
                                 number_of_classes, classifier_activation)

    predicted_label, acc = [], []
    data_processor.test_data_df[PsU.col_predicted_labels] = None

    for i, texts, labels in data_processor.get_test_data():
        predicted_label_, acc_ = model.do_prediction(texts, labels)
        print("[{}]Test Progress - {}/{} -- Accuracy: {:.4f}".format(i, i + 1,
                                                                     data_processor.test_batch_count,
                                                                     acc),
              end="\r", flush=True)
        predicted_label.append(predicted_label_)
        acc.append(acc_)

    acc = sum(acc) / len(acc)
    print("\nFinal Test Scores -- Accuracy - {:.3f}".format(acc))

    data_processor.test_data_df[PsU.col_predicted_labels] = None
    for i, label in enumerate(predicted_label):
        data_processor.test_data_df.at[i, PsU.col_predicted_labels] = label

    data_processor.test_data_df.to_csv(PsU.get_output_write_path(model_name, PsU.file_test_output))

    return predicted_label, acc


def add_params(parser):
    parser.add_argument(
        '--model_name',
        type=str,
        default="roberta-base",
        help='name_or_directory for pretrained model.')

    parser.add_argument(
        '--training',
        type=bool,
        default=False,
        help="Start model training.")

    parser.add_argument(
        '--testing',
        type=bool,
        default=False,
        help="Evaluate model on test data.")

    parser.add_argument(
        '--restart_training',
        type=bool,
        default=False,
        help="To restart training from last checkpoints")

    parser.add_argument(
        '--epochs',
        type=int,
        default=-1,
        help='Number of epoch model to trained')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=-1,
        help='Batch size to be used for data processing/training.')

    return parser


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_params(arg_parser)
    FLAGS, _ = arg_parser.parse_known_args()

    training_mode, evaluation_mode, restart_training_ = FLAGS.training, FLAGS.testing, FLAGS.restart_training

    model_name_ = FLAGS.model_name
    model_config_ = PsU.find_and_read_config(model_name_)

    epochs_ = FLAGS.epochs if FLAGS.epochs > 0 else model_config_[PsU.key_epochs]
    batch_size_ = FLAGS.batch_size if FLAGS.batch_size > 0 else model_config_[PsU.key_batch_size]

    if training_mode:
        train_model(model_name_, model_config_, epochs_, batch_size_, restart_training_)

    if evaluation_mode:
        test_model(model_name_, model_config_, batch_size_)
