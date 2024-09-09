from common import config
from entity.MyResNet import MyResNet 
import torch
from PIL import Image
from torchvision import transforms

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class Classify:
    def __init__(self):
        self.modelName = config.CLASSIFY_MODEL
        self.classes = config.CLASSES
        self.model = MyResNet()
        self.model.load_state_dict(torch.load(self.modelName))
        self.model.eval()

    def predict(self,imgPath,box):
        img = Image.open(imgPath).convert('RGB')
        crop_img = img.crop(box)
        final_img = transformer(crop_img)
        output = self.model(final_img.unsqueeze(0))
        probs = torch.softmax(output, dim=1) # softmax归一化为概率
        res = {self.classes[i] : probs[0][i].item() for i in range(len(self.classes))}
        # res按值排序
        res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        return res[0]
        