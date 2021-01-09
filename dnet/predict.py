import torch
from dnet.model import resnet50
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.Grayscale(1),
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, ], [0.229, ])])

# load image
# img = Image.open(r"C:\Users\admin\Desktop\新建文件夹\NORMAL (21).png")
# img = Image.open(r"C:\Users\admin\Desktop\新建文件夹\NORMAL (44).png")
# img = Image.open(r"C:\Users\admin\Desktop\新建文件夹\NORMAL (48).png")
# img = Image.open(r"C:\Users\admin\Desktop\新建文件夹\Viral Pneumonia (11).png")
# img = Image.open(r"C:\Users\admin\Desktop\新建文件夹\Viral Pneumonia (14).png")
# img = Image.open(r"C:\Users\admin\Desktop\新建文件夹\Viral Pneumonia (25).png")
# img = Image.open(r"C:\Users\admin\Desktop\新建文件夹\COVID-19(147).png")
# img = Image.open(r"C:\Users\admin\Desktop\新建文件夹\COVID-19(151).png")
img = Image.open(r"C:\Users\admin\Desktop\新建文件夹\103.png")



plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = resnet50(num_classes=3)
# load model weights
model_weight_path = "./resNet18new.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].numpy())
plt.show()
