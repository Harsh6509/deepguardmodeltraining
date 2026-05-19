import torch
from torchvision import transforms
from PIL import Image
from models.hybrid_model import HybridDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = HybridDetector().to(device)
model.load_state_dict(torch.load("deepguard_model.pth", map_location=device))
model.eval()

# Same transforms as training
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# hamne yaha pe classes define ki hai, agar tumne training ke time pe classes ko alag order me rakha hai to tum yaha pe bhi usi order me classes ko define karna, warna prediction galat aa sakta hai, lekin maine training ke time pe 'ai_generated' ko 0 aur 'real' ko 1 rakha tha to tum bhi yahi order follow karo,
classes = ['ai_generated', 'real']

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    pred = torch.argmax(output, 1).item()
    confidence = torch.softmax(output, dim=1)[0][pred].item()

    print(f"Prediction: {classes[pred]}")
    print(f"Confidence: {confidence:.2f}")

# yaha jo test imgage load hogi jipe model test karega
predict_image("test.jpg")