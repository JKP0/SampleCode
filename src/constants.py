
class GlobalPathsConstants:
    dir_data = "datasets"
    dir_model = "models"
    dir_output = "outputs"
    dir_training = "training"
    dir_best = "final_keras_model"
    dir_configs = "configs"


GpC = GlobalPathsConstants


class DataReadConstants:
    file_data_readme = "README.txt"
    file_data_train_val = "train_val.csv"
    file_data_test = "test.csv"

    data_col_id = "ID"
    data_col_tweet = "tweet"
    data_col_labels = "labels"

    col_id_add_txt = "t"

    add_class_others = "others"
    add_col_train_label = "train_label"


DrC = DataReadConstants


class ModelConstants:
    model_name_roberta_sentiment_latest = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model_name_roberta_sentiment = "cardiffnlp/twitter-roberta-base-sentiment"
    model_name_xlm_roberta_sentiment = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    dir_pre_trained_assets = "pre_trained_assets"
    dir_final_keras_model = "final_keras_model"
    model_ckpt_name = "model_state_{}_{}.ckpt"
    model_h5_name = "tf_model.h5"


McS = ModelConstants


class ProcessConstants:
    key_training = "training"
    key_validation = "validation"

    key_loss = "loss"
    key_acc = "accuracy"

    key_model_name = "model_name"
    key_pre_classifier_fc_units = "pre_classifier_fc_units"
    key_pre_classifier_activation = "pre_classifier_activation"
    key_number_of_classes = "number_of_classes"
    key_classifier_activation = "classifier_activation"

    key_epochs = "epochs"
    key_batch_size = "batch_size"
    key_validation_fraction = "validation_fraction"
    key_training_monitor = "training_monitor"
    key_learning_rate = "learning_rate"
    key_loss_name = "loss_name"

    text_val = "val"
    text_acc = "acc"
    text_loss = key_loss

    file_checkpoint = "checkpoint"
    file_ckpt_key = "model_checkpoint_path: "
    file_all_ckpt_key = "all_model_checkpoint_paths: "
    file_history = "training_history.json"

    model_config_name = "final_{}_config.json"
    col_predicted_labels = "predicted_labels"

    file_test_output = "test_data_prediction_{}_model.csv"
    file_validation_output = "validation_data_prediction_{}_model.csv"


PsC = ProcessConstants
