from datasets.cifar10_lsb import CIFAR10LSB

dataset_path = r"C:\Users\user6\Downloads\CIFAR10_lsb data-20261105T081348Z-1-001\CIFAR10_lsb data"
train_dataset = CIFAR10LSB(dataset_path, train=True)
test_dataset = CIFAR10LSB(dataset_path, train=False)

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

image, label = train_dataset[0]

print("Image size:", image.size)
print("Image mode:", image.mode)
print("Label:", label)
print("Image array shape:", train_dataset.data.shape)

