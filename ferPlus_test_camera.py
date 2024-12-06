import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from networks.DDAM import DDAMNet

class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']

def load_model(model_path, num_head, device):
    model = DDAMNet(num_class=8, num_head=num_head)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_emotion(model, device, frame):
    # Convert to grayscale and resize to 112x112
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Add channel dimension to match model input
    frame = cv2.resize(frame, (112, 112))
    frame = np.expand_dims(frame, axis=-1)  # Add channel dimension
    frame = np.repeat(frame, 3, axis=-1)  # Repeat channels to make it compatible with model
    
    # Transform input frame
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Adjust mean for grayscale
                             std=[0.5, 0.5, 0.5])  # Adjust std for grayscale
    ])
    
    input_tensor = transform(frame).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output, _, _ = model(input_tensor)
        _, prediction = torch.max(output, 1)
    
    return class_names[prediction.item()]

def main():
    model_path = './checkpoints/head4rl0.001prepro/processed.pth'
    num_head = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, num_head, device)
    
    # Start camera capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Press 'q' to quit.")
    box_size = 224  # Larger bounding box size

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        height, width, _ = frame.shape
        
        # Calculate the bounding box coordinates
        x1 = width // 2 - box_size // 2
        y1 = height // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Align your face in the box", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Crop and predict emotion in real-time
        cropped_face = frame[y1:y2, x1:x2]
        emotion = predict_emotion(model, device, cropped_face)
        
        # Display the predicted emotion on the video feed
        cv2.putText(frame, f"Emotion: {emotion}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Emotion Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit the program
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

