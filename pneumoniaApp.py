import sys

if 'torch' in sys.modules:
    import torch
    import types
    if isinstance(torch.classes, types.ModuleType):
        try:
            del torch.classes.__path__
        except AttributeError:
            pass
import torch
import streamlit as st  
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
st.set_page_config(page_title="Pneumonia Detection", page_icon=":guardsman:", layout="wide")
st.title("Pneumonia Detection using Transfer Learning")
st.subheader("This application uses a pre-trained model to detect pneumonia in chest X-ray images.")
@st.cache_resource
def pneumonia_model():
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 500),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(500, 2)
    )
    model.load_state_dict(torch.load("pneumoina_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model
def filter_model():
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 500),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(500, 2)
    )
    model.load_state_dict(torch.load("xray_filter.pth", map_location=torch.device('cpu')))
    model.eval()
    return model
img_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.Grayscale(num_output_channels=1),   
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5] )
    ])

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img,caption='uploaded image',width=250)
    if st.button('dectect',type='primary'):
        img = img_transforms(img)
        img = torch.unsqueeze(img, 0)
        try:
            filter_model = filter_model()
            with torch.no_grad():
                filter_output= filter_model(img)
                fil_prediction = torch.argmax(filter_output, dim=1).item()
                if fil_prediction == 1:
                    st.warning("This is not a valid chest X-ray image. Please upload a valid chest image.")
        except Exception as e:
            st.error(f"Error in filter model: {e}")
        
        try:
                if fil_prediction == 0:
                    st.write("This is a valid chest X-ray image. Proceeding with pneumonia detection.")
                    pneumonia_model = pneumonia_model()
                    with torch.no_grad():
                        pneumonia_output = pneumonia_model(img)
                        prediction = torch.argmax(pneumonia_output, dim=1).item()
                        labels = ['Normal', 'Pneumonia']   
                        if labels[prediction]=='Normal':  
                            st.success('This patient does not have pneumonia')
                        else:
                            st.warning('This patient has pneumonia')
        except Exception as e:
                st.error(f"Error in pneumonia model: {e}")

else:
    st.subheader("Please upload a chest X-ray image to make a prediction.")
