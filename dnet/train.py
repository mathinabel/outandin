import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
#import torchvision.model.dnet
from dnet.model import resnet34, resnet101,resnet18,resnet50,resnet152


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = {
        "train1": transforms.Compose([transforms.Grayscale(1),
                                     transforms.RandomResizedCrop(244),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, ], [0.229, ])]),
        "val1": transforms.Compose([transforms.Grayscale(1),
                                   transforms.Resize(256),
                                   transforms.CenterCrop(244),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, ], [0.229, ])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "C:"))
    # image_path = data_root + "/Users/admin/Desktop/新建文件夹/"
    image_path = "C:/Users/admin/Desktop/新建文件夹/"

    train_dataset = datasets.ImageFolder(root=image_path + "train1",
                                         transform=data_transform["train1"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=3)

    validate_dataset = datasets.ImageFolder(root=image_path + "val1",
                                            transform=data_transform["val1"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=3)

    net = resnet18(num_classes=3)
    # net = resnet34(num_classes=3)
    # net = resnet50(num_classes=3)
    # net = resnet101(num_classes=3)
    # net = resnet152(num_classes=3)

    # load pretrain weights
    # model_weight_path = "./resnet34-pre.pth"
    # missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
    # for param in net.parameters():
    #     param.requires_grad = False
    # change fc layer structure
    # inchannel = net.fc.in_features
    # net.fc = nn.Linear(inchannel, 3)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    best_acc = 0.0
    save_path = './resNet18new.pth'
    for epoch in range(100):
        # train
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            images = images.to(device)
            logits = net(images)
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print train process
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
        print()

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model only have last output layer
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / val_num
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                  (epoch + 1, running_loss / step, val_accurate))

    print('Finished Training')
