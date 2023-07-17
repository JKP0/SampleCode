from utils import PsU, DisplayProgress


class TrainingProcessor(object):
    """
    Custom Trainer
    """

    def _load_last_check_point(self):
        model_checkpoint, all_model_checkpoint, epoch = PsU.get_model_checkpoint_file_name(self.model.model_name)

        try:
            assert self.model.load_intermediate_weights(PsU.make_training_ckpt_path(self.model.model_name,
                                                                                    model_checkpoint))
            print("Loaded Model From \nCheckpoint - {} \nAll Checkpoint - {}".format(model_checkpoint,
                                                                                     all_model_checkpoint))
        except AssertionError:
            print("Weight Load Failed.\nStarting Training From Epoch Zero-0.")
            epoch = 0

        return epoch

    def _validate_and_save(self, epoch):
        current_monitor = self.history_data[self.monitor_key[0]][self.monitor_key[1]][-1]

        if ((PsU.text_acc in self.monitor_key[1] and current_monitor > self.last_monitor_value) or
                (PsU.text_loss in self.monitor_key[1] and current_monitor < self.last_monitor_value)):
            self.model.save_model((self.log_dir, self.model.model_name, epoch, round(current_monitor, 3)))

        PsU.log_json_data(self.model.model_name, self.history_data)

        return True

    def _update_history(self, training_data, validation_data, epoch, dspl):
        loss, acc = training_data
        val_loss, val_acc = validation_data

        loss, acc = sum(loss), sum(acc) / len(acc)
        val_loss, val_acc = sum(val_loss), sum(val_acc) / len(val_acc)

        self.history_data[PsU.key_training][PsU.key_loss].append(loss)
        self.history_data[PsU.key_training][PsU.key_acc].append(acc)

        self.history_data[PsU.key_validation][PsU.key_loss].append(val_loss)
        self.history_data[PsU.key_validation][PsU.key_acc].append(val_acc)

        dspl.display(dspl.print_log_format.format(epoch + 1, loss, acc, val_loss, val_acc),
                     end="\n", flush=False, flush_all_counters=True)

        return True

    def train(self, start_epoch=0):
        if start_epoch != 0:
            start_epoch = self._load_last_check_point()

        dspl = DisplayProgress(self.verbose)

        for epoch in range(start_epoch, self.epochs):
            dspl.display(dspl.print_epoch_progress.format(epoch, epoch + 1, self.epochs))

            #  Training Step
            loss, acc = [], []
            for i, texts, labels in self.data_processor.get_training_data():
                loss_, acc_ = 0, 0
                dspl.display(dspl.print_training_batch_progress.format(
                    i, i + 1, self.data_processor.training_batch_count, loss_, acc_))
                loss_, acc_ = self.model(texts, labels, training=True)
                loss.append(loss_)
                acc.append(acc_)

            dspl.display("", flush=False, in_line_after_last=True)

            #  Validation Step
            val_loss, val_acc = [], []
            for i, texts, labels in self.model.get_validation_data():
                loss_, acc_ = 0, 0
                dspl.display(dspl.print_validation_batch_progress.format(
                    i, i + 1, self.data_processor.validation_batch_count, loss_, acc_))
                loss_, acc_ = self.model(texts, labels)
                val_loss.append(loss_)
                val_acc.append(acc_)

            # dspl.display("", flush=False, in_line_after_last=True)
            self._update_history((loss, acc), (val_loss, val_acc), epoch, dspl)
            self._validate_and_save(epoch)

        return True

    def __init__(self, data_processor, training_model, epochs, monitor_key="val_acc", verbose=True):
        self.data_processor = data_processor
        self.model = training_model
        self.epochs = epochs
        self.log_dir = PsU.training_log_dir

        self.history_data = {PsU.key_training: {PsU.key_loss: [], PsU.key_acc: []},
                             PsU.key_validation: {PsU.key_loss: [], PsU.key_acc: []}}

        self.last_monitor_value = 99999 if PsU.text_loss in monitor_key else 0

        self.monitor_key = [PsU.key_validation if PsU.text_val in monitor_key else PsU.key_training,
                            PsU.key_acc if PsU.text_acc in monitor_key else PsU.key_loss]
        self.verbose = verbose
