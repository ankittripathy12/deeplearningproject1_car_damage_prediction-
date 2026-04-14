import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

trained_model = None

class_names = [
    'Front Breakage', 'Front Crushed', 'Front Normal',
    'Rear Breakage', 'Rear Crushed', 'Rear Normal'
]

# ✅ Model Class
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        # Freeze layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace FC layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ✅ Prediction Function (FIXED)
def predict(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except:
        return "Invalid Image", 0.0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0)

    global trained_model

    if trained_model is None:
        trained_model = CarClassifierResNet()

        # ✅ Safe model path (Render compatible)
        BASE_DIR = Path(__file__).resolve().parent
        MODEL_PATH = BASE_DIR / "model" / "saved_model.pth"

        trained_model.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        )

        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)

        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

        return class_names[predicted_class.item()], confidence.item()
