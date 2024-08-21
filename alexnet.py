from torchvision import transforms
from torchvision import datasets
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(
        data_dir,
        batch_size,
        augment,
        random_seed,
        valid_size=0.1,  # Разделяем набор данных на train 90% и test 10%. Разделение происходит стохастически.
        shuffle=True
):
    """
    Загрузка и подготовка набора данных для обучения
    :param data_dir:
    :param batch_size:
    :param augment:
    :param random_seed:
    :param valid_size:
    :param shuffle:
    :return:
    """
    # Нормализуем данные с картинки каждого канала (красного, зеленого и синего).
    # Среднее арифметическое и стандартное отклонение для своего набора данных нужно рассчитать вручную, но для CIFAR-10 они есть в сети.
    normalize = transforms.Normalize(
        mean=[0.49139968, 0.48215827, 0.44653124],
        std=[0.24703233, 0.24348505, 0.26158768],
    )
    valid_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])

    # Расширяем набор данных за счет аугументации. Применяем аугументацию только к подмножеству для обучения.
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ])

    # собственно выгрузка, если локально данных нет
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )
    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    # добавляем стохастичность для формирования train и valid выборок
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Делим train данные на на наборы (batches) таким образом, чтобы в каждом вероятность присутствия каждого класса была такой же как и во всей train выборке
    # Используем torch loaders для работы с набором данных итеративно и быстро
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader


def get_test_loader(
        data_dir,
        batch_size,
        shuffle=True
):
    """
    Загрузка и подготовка набора данных для тестирования
    :param data_dir:
    :param batch_size:
    :param shuffle:
    :return:
    """
    # Нормализуем данные с картинки каждого канала (красного, зеленого и синего).
    # Среднее арифметическое и стандартное отклонение для своего набора данных нужно рассчитать вручную, но для CIFAR-10 они есть в сети.
    normalize = transforms.Normalize(
        mean=[0.49139968, 0.48215827, 0.44653124],
        std=[0.24703233, 0.24348505, 0.26158768],
    )
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])

    # собственно выгрузка, если локально данных нет
    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    # Делим test данные на на наборы (batches) таким образом, чтобы в каждом вероятность присутствия каждого класса была такой же как и во всей test выборке
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader


#     Finally, our last layer outputs 10 neurons which are our final predictions for the 10 classes of objects

class AlexNet(nn.Module):
    """
    Модели наследуются от nn.Module.
    """

    def __init__(self, num_classes=10):
        """
        Объявляются сами слои с размерностью.
        :param num_classes:
        """
        super(AlexNet, self).__init__()

        # первыми идут сверточные слои nn.Conv2D
        # с размером ядра и каналами ввода/вывода,
        # нормировкой выходных карт активации,
        # relu для нелинейности,
        # на некоторых слоях применяем к картам активации max pooling для сжатия/концентрации данных
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        # затем разворачиваем последний сверточный слой в первый вектор слой для полносвязной сети из трех слоев
        # зануляем дропаутом 50% весов на каждом слое в момент прямого шага на этапе обучения на train выборке
        # relu для нелинейности везде кроме последнего выходного слоя
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())

        # последний слой выводит 10 нейронов, которые являются нашими окончательными предсказаниями для 10 классов объектов
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        """
        Последовательность слоев при обработке изображения
        :param x:
        :return:
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    # выбор устройства для расчета
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 10
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.005

    # подготовка лодеров для train и test выборок
    train_loader, valid_loader = get_train_valid_loader(data_dir='./data', batch_size=64, augment=False, random_seed=1)
    test_loader = get_test_loader(data_dir='./data', batch_size=64)

    # определение устройства для расчета модели
    model = AlexNet(num_classes).to(device)

    # критерий сходимости (кроссэнтропия для классификации)
    criterion = nn.CrossEntropyLoss()

    # выбор оптимизатора градиентного спуска
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0.005, betas=(0.9, 0.999))

    # собственно обучение
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # выбор устройства для расчетов
            images = images.to(device)
            labels = labels.to(device)

            # прямой шаг
            outputs = model(images)
            loss = criterion(outputs, labels)

            # обратный шаг и оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # валидация получившейся модели на valid_loader
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

        # валидация получившейся модели на test_loader
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} test images: {} %'.format(5000, 100 * correct / total))
