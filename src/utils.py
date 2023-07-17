import os
import json

from constants import GpC, DrC, McS, PsC


class GlobalPathsUtils(GpC):
    PROJECT_BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
    DATA_DIR = os.path.join(PROJECT_BASE, GpC.dir_data)
    MODEL_DIR = os.path.join(PROJECT_BASE, GpC.dir_model)
    TRAINING_HISTORY_DIR = os.path.join(PROJECT_BASE, GpC.dir_training)


GpU = GlobalPathsUtils


class DataReadUtils(GpU, DrC):
    dir_dataset = GpU.DATA_DIR


DrU = DataReadUtils


class ModelUtils(GpU, McS):
    pre_train_model_dir = os.path.join(GpU.MODEL_DIR, McS.dir_pre_trained_assets)
    final_model_save_dir = os.path.join(GpU.MODEL_DIR, McS.dir_final_keras_model)
    training_log_dir = GpU.TRAINING_HISTORY_DIR

    @classmethod
    def make_model_ckpt_path(cls, model_name, epoch, acc):
        path_im = os.path.join(cls.training_log_dir, model_name)
        if not os.path.exists(path_im):
            os.makedirs(path_im)

        return os.path.join(model_name, cls.model_ckpt_name.format(epoch, acc))

    @classmethod
    def make_model_h5_save_path(cls, model_name):
        path_im = os.path.join(McS.final_model_save_dir, model_name)
        if not os.path.exists(path_im):
            os.makedirs(path_im)

        return os.path.join(path_im, cls.model_h5_name)


MlU = ModelUtils


class ProcessUtils(GpU, PsC):
    training_log_dir = GpU.TRAINING_HISTORY_DIR
    final_model_h5_dir = os.path.join(GpU.MODEL_DIR, GpC.dir_best)
    final_model_h5_file = McS.model_h5_name

    @classmethod
    def log_json_data(cls, model_name, history_data):
        path = os.path.join(cls.training_log_dir, model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, cls.file_history)

        with open(path, "w") as jf:
            json.dump(jf, history_data, indent=3)

        return True

    @classmethod
    def get_training_checkpoint_file(cls, model_name):
        return os.path.join(cls.training_log_dir, model_name, cls.file_checkpoint)

    @classmethod
    def get_model_checkpoint_file_name(cls, model_name):
        path = cls.get_training_checkpoint_file(model_name)

        with open(path) as txt:
            data = txt.readlines()

        model_checkpoint = data[0].replace(PsC.file_ckpt_key, "").replace('"', '').replace("\n", "").strip(" ")
        all_model_checkpoint = data[1].replace(PsC.file_ckpt_key, "").replace('"', '').replace("\n", "").strip(" ")

        epoch = int(model_checkpoint.split("_")[-2])

        return model_checkpoint, all_model_checkpoint, epoch

    @classmethod
    def make_training_ckpt_path(cls, model_name, ckpt_file_name):
        return os.path.join(cls.training_log_dir, model_name, ckpt_file_name)

    @classmethod
    def get_final_model_h5_path(cls, model_name):
        return os.path.join(cls.final_model_h5_dir, model_name, cls.final_model_h5_file)

    @classmethod
    def get_final_model_ckpt_path(cls, model_name):
        ckpt_file_name, _, epoch = cls.get_model_checkpoint_file_name(model_name)

        return cls.make_training_ckpt_path(model_name, ckpt_file_name), epoch

    @classmethod
    def get_model_config_path(cls, model_name):
        model_name = model_name.replace("/", "--")
        return os.path.join(cls.MODEL_DIR, GpC.dir_configs,
                            cls.model_config_name.format(model_name))

    @classmethod
    def find_and_read_config(cls, model_name):
        json_path = cls.get_model_config_path(model_name)

        with open(json_path) as jf:
            config = json.load(jf)

        return config

    @classmethod
    def get_output_write_path(cls, model_name, file_name):
        return os.path.join(cls.DATA_DIR, GpC.dir_output,
                            file_name.format(model_name.replace("/", "--")))


PsU = ProcessUtils


class DisplayProgress:
    print_log_format = "\n[{}]- Training - Loss: {:.4f}, Accuracy: {:.4f}; Validation - Loss: {:.4f}, Accuracy: {:.4f}"
    print_training_batch_progress = "[{}]-TrainingSteps - {}/{} - Loss: {:.3f} - Accuracy: {:.3f}"
    print_validation_batch_progress = "[{}]-ValidationSteps - {}/{} - Loss: {:.3f} - Accuracy: {:.3f}"
    print_epoch_progress = "\n[{}] Epochs - {}/{}"

    def display(self, text, end="\r", flush=True, in_line_after_last=False, flush_all_counters=False):
        if self.verbose:
            if flush_all_counters:
                self.last_text = ""
                self.last_text_use = []

            elif in_line_after_last and flush:
                self.last_text_use = [self.last_text + "\t" + text]

            elif in_line_after_last:
                self.last_text_use.append(self.last_text + "\t" + text)

            self.last_text = text

            text = " ".join(self.last_text_use) + text

            print(text, end=end, flush=flush)

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.last_text = ""
        self.last_text_use = []
