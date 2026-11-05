
import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CIFAR10LSB(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if train:
            batch_files = [
                "data_batch_1_lsb",
                "data_batch_2_lsb",
                "data_batch_3_lsb",
                "data_batch_4_lsb",
                "data_batch_5_lsb"
            ]
        else:
            batch_files = ["test_batch_lsb"]

        data_list = []
        labels_list = []

        for filename in batch_files:
            file_path = os.path.join(root, filename)

            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Dataset file not found: {file_path}"
                )

            with open(file_path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")

            data = batch.get(b"data", batch.get("data"))
            labels = batch.get(b"labels", batch.get("labels"))

            if data is None or labels is None:
                raise ValueError(f"Invalid CIFAR-10 batch: {filename}")

            data_list.append(data)
            labels_list.extend(labels)

        self.data = np.concatenate(data_list, axis=0)

        # Convert CIFAR-10 channel-first data into HWC images.
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose(0, 2, 3, 1)

        self.targets = labels_list

        self.classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, target