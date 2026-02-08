
import warnings
warnings.filterwarnings('ignore')
import torch
import torchvision.models as models
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    exit(1)

# Dummy Model
model = models.densenet121(pretrained=False)
target_layers = [model.features[-1]]

# Dummy Input
input_tensor = torch.rand(1, 3, 224, 224)

# Initialize GradCAM
try:
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    print("GradCAM computation successful.")
    print(f"CAM shape: {grayscale_cam.shape}")
except Exception as e:
    print(f"GradCAM failed: {e}")
    exit(1)
