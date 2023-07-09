import os
import pandas as pd

from constants import DrC


class DataInitialView(object):
    def __assign_labels_toread(self, data_dir):
        readme_path = os.path.join(data_dir, DrC.file_data_readme)

        tag_catch = "List of the 12 different vaccine concerns in the datasets"

        with open(readme_path) as txt:
            text_contents = txt.readlines()

        read_not_class = True
        label_id = 0

        for txt_line in text_contents:
            if read_not_class or label_id >= 12:
                read_not_class = tag_catch not in txt_line
                continue
            start_index = txt_line.index("[")
            end_index = txt_line.index("]")
            label_name = txt_line[start_index + 1: end_index].strip().lower()

            self.labels_to_read[label_id] = label_name

            label_id += 1

        self.labels_to_read[label_id] = DrC.add_class_others

        return True

    def __update_label_dist(self):
        labels_dist = {self.labels_to_read[e]: 0 for e in self.labels_to_read}

        for e, v in self.data_df.labels.value_counts().to_dict().items():
            if e in labels_dist:
                labels_dist[e] = v

            else:
                labels_dist[DrC.add_class_others] = labels_dist[DrC.add_class_others] + v

        labels_dist = sorted(labels_dist.items(), key=lambda e: e[1], reverse=True)
        return dict(labels_dist)

    def __assign_trainable_label(self):
        self.data_df[DrC.add_col_train_label] = None

        for i in range(self.data_df.shape[0]):
            current_label = self.data_df.at[i, DrC.data_col_lables]

            if current_label.strip().lower() in self.labels_to_read.values():
                self.data_df.at[i, DrC.add_col_train_label] = current_label.strip().lower()
            else:
                self.data_df.at[i, DrC.add_col_train_label] = DrC.add_class_others

        return True

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = DrC.drive_project_dir
        self.data_path = os.path.join(data_dir, DrC.file_data_train_val)

        self.data_df = pd.read_csv(self.data_path)

        print("ColumnsInData", self.data_df.columns)

        self.labels_to_read = {}
        self.__assign_labels_toread(data_dir)
        print("Labels To Be Read", self.labels_to_read)

        self.labels_dist = self.__update_label_dist()

        self.__assign_trainable_label()


if __name__ == "__main__":
    DiV = DataInitialView()
    # print("DataFrame", DiV.data_df)
    print("Data Labels Count\n", DiV.data_df.labels.value_counts())
    print("Labels Distribution\n", DiV.labels_dist)
    print("Labels Distribution\n", DiV.labels_dist)
    print("Total Samples Validated", sum([DiV.labels_dist[k] for k in DiV.labels_dist]))
    print("Training Labels Count:\n", DiV.data_df.train_label.value_counts())
