import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import time


def plot_confusion_matrix(cm, classes, save_path):
    """
    Plots the confusion matrix and saves it to an image file.
    """
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    # Save the confusion matrix as an image
    plt.savefig(save_path)
    plt.close()


# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
# Reshape the data to (2866, 64, 561)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# Convert the data to PyTorch tensors and move to the GPU
X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Tree-based classifiers
classifiers = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42)
]

for classifier in classifiers:
    print(f"Training {classifier.__class__.__name__}...")
    start_time = time.time()
    classifier.fit(X_train_reshaped, y_train)
    end_time = time.time()

    # Predict using the classifier
    y_train_pred = classifier.predict(X_train_reshaped)
    y_test_pred = classifier.predict(X_test_reshaped)

    # Calculate confusion matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    print(f"{classifier.__class__.__name__} - Confusion Matrix (Train):")
    print(cm_train)
    print(f"{classifier.__class__.__name__} - Confusion Matrix (Test):")
    print(cm_test)

    # Calculate precision, recall, train accuracy, and test accuracy
    precision_train = precision_score(y_train, y_train_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_train = recall_score(y_train, y_train_pred)
    recall_test = recall_score(y_test, y_test_pred)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    print(f"{classifier.__class__.__name__} - Precision (Train): {precision_train}")
    print(f"{classifier.__class__.__name__} - Precision (Test): {precision_test}")
    print(f"{classifier.__class__.__name__} - Recall (Train): {recall_train}")
    print(f"{classifier.__class__.__name__} - Recall (Test): {recall_test}")
    print(f"{classifier.__class__.__name__} - Accuracy (Train): {accuracy_train}")
    print(f"{classifier.__class__.__name__} - Accuracy (Test): {accuracy_test}")

    # Plot confusion matrix and save it
    plot_confusion_matrix(cm_test, classes=['Class 0', 'Class 1'], save_path=f"{classifier.__class__.__name__}_confusion_matrix.png")

    print(f"Time taken for training {classifier.__class__.__name__}: {end_time - start_time} seconds")
    print("---------------------------------------------------")

# Neural Network classifier
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(X_train_tensor.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training Neural Network...")
start_time = time.time()

# Training loop
num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

end_time = time.time()

# Predict using the trained model
y_train_pred = torch.argmax(model(X_train_tensor), dim=1).cpu().numpy()
y_test_pred = torch.argmax(model(X_test_tensor), dim=1).cpu().numpy()

# Calculate confusion matrix
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)
print("Neural Network - Confusion Matrix (Train):")
print(cm_train)
print("Neural Network - Confusion Matrix (Test):")
print(cm_test)

# Calculate precision, recall, train accuracy, and test accuracy
precision_train = precision_score(y_train, y_train_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_train = recall_score(y_train, y_train_pred)
recall_test = recall_score(y_test, y_test_pred)
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print("Neural Network - Precision (Train):", precision_train)
print("Neural Network - Precision (Test):", precision_test)
print("Neural Network - Recall (Train):", recall_train)
print("Neural Network - Recall (Test):", recall_test)
print("Neural Network - Accuracy (Train):", accuracy_train)
print("Neural Network - Accuracy (Test):", accuracy_test)

# Plot confusion matrix and save it
plot_confusion_matrix(cm_test, classes=['Class 0', 'Class 1'], save_path='NeuralNetwork_confusion_matrix.png')

print(f"Time taken for training Neural Network: {end_time - start_time} seconds")
print("---------------------------------------------------")
