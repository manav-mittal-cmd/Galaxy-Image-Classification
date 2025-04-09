#import necessary libraries
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import pandas as pd
import timm


# Header stuff and basic info
st.title("Galaxy Classification Frontend")
st.write("Purpose of this website is to classify images of galaxies based on their morphological characteristics")
st.write("Currently we can classify images of size 227x227 into 3 types: Spiral (S), Elliptical (E) and Spiral Barred (SB)")
st.write("Our model's accuracy is 86.59%")

# prompt user to enter an image
uploaded_file = st.file_uploader("Upload a galaxy image...", type = ["jpg", "jpeg", "png"])

# if image was uploaded
if uploaded_file is not None:
    # Open the image using PIL library
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.success("Image uploaded successfully!")
else:
    st.error("Image not uploaded")


# resize the image and apply appropriate transformations
transform = transforms.Compose({
    transforms.Resize((227,277)),
    transforms.ToTensor(),
})

# Load and preprocess the image
def preprocess_image(image, transform):
    image = image.convert("RGB")  # Ensure it's in RGB format
    image_tensor = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    return image_tensor

# Apply preprocessing and display the transformed image
if uploaded_file is not None:
    image_tensor = preprocess_image(image, transform)

    # Display the transformed image
    st.image(image_tensor.squeeze(0).permute(1, 2, 0).numpy(), caption="Transformed Image", use_container_width=True)


# Define model
class GalaxyImageClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(GalaxyImageClassifier, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        # removing the last layer of the model
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Linear(enet_out_size, num_classes)


    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
       # x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


# Load the trained model
model = GalaxyImageClassifier(num_classes=3)
model.load_state_dict(torch.load('./galaxy_classifier.pth', map_location=torch.device('cpu')))

# set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Predict using the model
def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        # converts the raw data from our model(logits) into a probability distribution across classes
        # uses sotftmax function
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # converts probabilities tensor into a numpy array
    return probabilities.cpu().numpy().flatten()

predicted_class = predict(model, image_tensor)

#print(f'Predicted Class: {predicted_class}')

# display predictions
def visualize_predictions(probabilities, class_names, true_label=None):

    # Display true label if available
    if true_label is not None:
        st.write(f"True Label: {true_label}")

    # Display predictions as a bar chart
    prediction_data = {
        "Class Names": list(class_names.values()),
        "Probabilities": probabilities
    }
    df = pd.DataFrame(prediction_data)

    st.bar_chart(df.set_index("Class Names")["Probabilities"])

    # Display a detailed table of predictions
    st.write("Class Probabilities:")
    st.dataframe(df)


# display final result
class_names = {0: 'E', 1: 'S', 2: 'SB'}

visualize_predictions(predicted_class, class_names)


