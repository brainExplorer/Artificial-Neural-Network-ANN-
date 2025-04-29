"""
Problem tanimi: mnist veri seti kullanarak rakam siniflandirma
MNIST
ANN: Artificial Neural Network
"""

# %% Importing libraries
import torch            # PyTorch library for deep learning, tensor computation
import torch.optim as optim  # Optimizers for training models
import torch.nn as nn # Neural network module for building models
import torchvision # Computer vision library for datasets and transformations
import torchvision.transforms as transforms # Transformations for data preprocessing
import matplotlib.pyplot as plt # Library for data visualization

# optional: set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

60000/64
# data loading and preprocessing
def get_data_loaders(batch_size=64):  # batch size is the number of samples processed before the model is updated
    """
    Function to load and preprocess the MNIST dataset.
    Args:
        batch_size (int): Batch size for data loaders.
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
    """
    # Define transformations for the training and testing sets
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images , pixels to range [-1, 1]
    ])

    # Load the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders for training and testing sets
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#train_loader, test_loader = get_data_loaders()

# data visualization
def visuliaze_samples(loader,n):
    """
    Function to visualize a batch of samples from the data loader.
    Args:
        loader (DataLoader): DataLoader for the dataset.
        n (int): Number of samples to visualize.
    """
    # Get a batch of data
    images, labels = next(iter(loader))

    fig, axes = plt.subplots(1, n, figsize=(10, 5))    # Create a figure with subplots
    for i in range(n):
        ax = axes[i]  # Get the current axis
        ax.imshow(images[i].squeeze(), cmap='gray')  # Display the image in grayscale
        ax.set_title(f'Label: {labels[i].item()}')  # Set the title to the label of the image
        ax.axis('off')  # Hide the axis     
    plt.show()  # Show the plot
    
#visuliaze_samples(train_loader, 4)  # Visualize 4 samples from the training set

# %% defineing the ann model

# class for the ANN model
class NeuralNetwork(nn.Module): # Inherit from nn.Module to create a custom neural network class
    def __init__(self): # Constructor method to initialize the model
        super(NeuralNetwork, self).__init__()  # Initialize the parent class
        
        self.flatten = nn.Flatten()  # Flatten layer to convert 2D images to 1D vectors
        
        self.fc1 = nn.Linear(28 * 28, 128)  # First fully connected layer (input: 28x28=784, output: 128)
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer (input: 128, output: 64)
        self.fc3 = nn.Linear(64, 10)  # Third fully connected layer (input: 64, output: 10 for digits 0-9)

    def forward(self, x): # Forward pass method to define the computation performed at every call, x is the input tensor
        x = self.flatten(x)  # Flatten the input tensor
        x = self.fc1(x)  # Pass through the first layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)  # Pass through the second layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc3(x)  # Pass through the third layer     
        return x # Return the output tensor


#creating the model and compiling it
#model = NeuralNetwork().to(device)  # Create an instance of the model and move it to the specified device (GPU or CPU)

# defineing the loss and optimizer
define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), # Define loss function (CrossEntropyLoss for multi-class classification)
    optim.Adam(model.parameters(), lr=0.001)  # Define optimizer (Adam) with learning rate of 0.001 
)

#criterion, optimizer = define_loss_and_optimizer(model)  # Unpack the loss function and optimizer

# %% training the model
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    
    # modelimizi egitim moduna aliyoruz
    model.train()  # Set the model to training mode
    train_loses = []  # List to store training losses for each epoch
    
    #belirtilen epoch sayisi kadar egitim yapilacak
    for epoch in range(epochs):
        total_loss = 0.0 # Initialize total loss for this epoch
        for images, labels in train_loader: # tum egitim verileri uzerindende iterasyon gerceklesitr
            images, labels = images.to(device), labels.to(device) # Move images and labels to the specified device (GPU or CPU)
            optimizer.zero_grad() # Zero the gradients of the optimizer before backpropagation
            predictions = model(images)
            loss = criterion(predictions, labels) # Compute the loss using the predictions and true labels
            loss.backward() # Backpropagation to compute gradients
            optimizer.step() # Update the model parameters using the optimizer
            total_loss += loss.item() # Accumulate the loss for this batch
        avg_loss = total_loss / len(train_loader) # Calculate the average loss for this epoch
        train_loses.append(avg_loss) # Append the average loss to the list
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.3f}")
    plt.figure()
    plt.plot(range(1, epochs + 1), train_loses, marker = "o", linestyle = "-", label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend()
    plt.show()
    
#train_model(model, train_loader, *define_loss_and_optimizer(model), epochs=10) # Train the model for 10 epochs
#train_model(model, train_loader, criterion, optimizer, epochs=1) # Train the model for 10 epochs
# %% testing the model

def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0  # Initialize correct predictions counter
    total = 0  # Initialize total predictions counter
    
    with torch.no_grad():  # Disable gradient calculation for testing
        for images, labels in test_loader:  # Iterate over the test data
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)  # Get model predictions
            _, predicted = torch.max(predictions.data, 1)  # Get the predicted class with the highest score
            total += labels.size(0)  # Update total predictions counter
            correct += (predicted == labels).sum().item()  # Update correct predictions counter
            
    print(f'Accuracy of the model on the test set: {100 * correct / total:.3f}%')  # Print the accuracy of the model
   
#test_model(model, test_loader)  # Test the model on the test set 

# %% main

if __name__ == "__main__":
    # Load data
    train_loader, test_loader = get_data_loaders()
    
    visuliaze_samples(train_loader, 5)  # Visualize 4 samples from the training set
    model = NeuralNetwork().to(device)  # Create an instance of the model and move it to the specified device (GPU or CPU)
    # Define loss function and optimizer
    criterion, optimizer = define_loss_and_optimizer(model)  # Unpack the loss function and optimizer
    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs=10)  # Train the model for 10 epochs
    # Test the model    
    test_model(model, test_loader)  # Test the model on the test set    
    