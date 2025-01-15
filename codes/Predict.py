import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# 定义数据预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



def load_class_labels(json_file):
    with open(json_file, 'r') as f:
        class_labels = json.load(f)
    return class_labels



def find_inpaint_v(folder_path):
    for root, dirs, files in os.walk(folder_path):
        if 'inpaint_v.png' in files:
            return os.path.join(root, 'inpaint_v.png')
    return None

def predict(folder_path):

    num_classes = 6  # modify according to your needs
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('final_model.pth')) # replace with your trained model
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')


    model.eval()


    image_path = find_inpaint_v(folder_path)


    json_file = 'your_json_file_path'  # replace with your json file path
    class_labels = load_class_labels(json_file)

    if image_path:
        # 加载图片
        image = Image.open(image_path).convert('RGB')

        # 预处理图片
        input_tensor = test_transform(image)
        input_batch = input_tensor.unsqueeze(0)

        # 将输入移动到合适的设备
        input_batch = input_batch.to('cuda' if torch.cuda.is_available() else 'cpu')

        # 进行预测
        with torch.no_grad():
            output = model(input_batch)
            _, predicted_idx = torch.max(output, 1)

        # 获取预测类别
        predicted_class = class_labels[str(predicted_idx.item())]

        # 打印预测结果
        print(f'Predicted class: {predicted_class}')

        # 可视化图片和预测结果
        plt.imshow(image)
        plt.title(f'Predicted Class: {predicted_class}')
        plt.axis('off')
        plt.show()
    else:
        print('inpaint_v.png not found in the specified folder.')