import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from torch.serialization import add_safe_globals
from config import USER_ID, ACTIONS

# Add LabelEncoder to safe globals for PyTorch loading
add_safe_globals([LabelEncoder])

class GestureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class GestureNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GestureNet, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, 
                           batch_first=True, dropout=0.2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step
        return self.classifier(lstm_out)

class GestureModelTrainer:
    def __init__(self):
        self.user_data_dir = os.path.join("gesture_data", USER_ID)
        os.makedirs(self.user_data_dir, exist_ok=True)
        self.model_path = os.path.join(self.user_data_dir, "gesture_model.pt")
        self.label_mapping_file = os.path.join(self.user_data_dir, "label_mapping.json")
        self.gesture_class_map_file = os.path.join(self.user_data_dir, "gesture_class_map.json")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Data directory: {self.user_data_dir}")
        
        # Load dataset info
        info_file = os.path.join(self.user_data_dir, "dataset_info.json")
        if not os.path.exists(info_file):
            raise FileNotFoundError(f"Dataset info file not found at {info_file}. Please run data collection first.")
            
        with open(info_file, 'r') as f:
            self.dataset_info = json.load(f)
        
        # Use centralized ACTIONS list
        self.actions = ACTIONS
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
        # Model parameters
        self.input_size = 63  # 21 landmarks * 3 coordinates
        self.hidden_size = 128
        
        # Training parameters
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 0.001
        
    def build_gesture_classes(self):
        """
        Returns:
            gesture_class_map: dict {gesture_class_index: [action_id, ...]}
            action_to_gesture_class: dict {action_id: gesture_class_index}
        """
        action_map = self.dataset_info.get("action_map", {})
        # Find unique gesture sources (actions that were actually recorded)
        gesture_sources = []
        for action_id, mapped_id in action_map.items():
            if mapped_id is not None and int(action_id) == mapped_id:
                gesture_sources.append(mapped_id)
        gesture_sources = sorted(set(gesture_sources))
        # Map each gesture source to a gesture class index
        gesture_class_map = {}
        action_to_gesture_class = {}
        for idx, src in enumerate(gesture_sources):
            gesture_class_map[idx] = []
            for action_id, mapped_id in action_map.items():
                if mapped_id == src:
                    gesture_class_map[idx].append(int(action_id))
                    action_to_gesture_class[int(action_id)] = idx
        return gesture_class_map, action_to_gesture_class

    def load_data(self):
        """Load and preprocess the gesture data, using action_map for shared gestures"""
        features = []
        labels = []
        gesture_class_map, action_to_gesture_class = self.build_gesture_classes()
        print("\nGesture class mapping:")
        for idx, acts in gesture_class_map.items():
            names = [dict(self.actions)[aid] for aid in acts]
            print(f"Class {idx}: actions {acts} ({' / '.join(names)})")
        print("\nLoading dataset...")
        for rel_path, info in self.dataset_info["samples"].items():
            action_id = info["action_id"]
            if action_id not in action_to_gesture_class:
                continue
            gesture_class = action_to_gesture_class[action_id]
            try:
                full_path = os.path.join(self.user_data_dir, rel_path)
                if not os.path.exists(full_path):
                    print(f"Warning: File not found: {full_path}")
                    continue
                data = np.load(full_path)
                features.append(data)
                labels.append(gesture_class)
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                continue
        if not features:
            raise ValueError("No valid data files found. Please check the dataset directory.")
        features = np.array(features)
        labels = np.array(labels)
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(labels)}")
        unique_labels = np.unique(labels)
        print(f"Unique gesture classes: {sorted(unique_labels)}")
        print("\nClass distribution:")
        for label in sorted(unique_labels):
            count = np.sum(labels == label)
            print(f"Class {label}: {count} samples")
        # Fit label encoder and transform labels
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        # Save label mapping
        label_mapping = {
            int(label): int(encoded_label) 
            for label, encoded_label in zip(labels, encoded_labels)
        }
        with open(self.label_mapping_file, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        print("\nLabel mapping:")
        for original, encoded in sorted(label_mapping.items()):
            print(f"Original label {original} -> Encoded label {encoded}")
        # Update number of classes
        self.num_classes = len(self.label_encoder.classes_)
        print(f"\nNumber of gesture classes: {self.num_classes}")
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        # Create data loaders
        train_dataset = GestureDataset(X_train, y_train)
        test_dataset = GestureDataset(X_test, y_test)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                     shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        # Save gesture class map for runtime
        with open(self.gesture_class_map_file, 'w') as f:
            json.dump({str(k): v for k, v in gesture_class_map.items()}, f, indent=2)
        return X_train.shape[1:]
    
    def train_model(self):
        """Train the gesture recognition model"""
        # Load and prepare data
        print("\nPreparing data...")
        input_shape = self.load_data()
        
        # Create model
        print("\nInitializing model...")
        self.model = GestureNet(self.input_size, self.hidden_size, 
                               self.num_classes).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        print("\nStarting training...")
        print(f"Total epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        
        # Training loop
        best_loss = float('inf')
        best_accuracy = 0.0
        train_losses = []
        test_losses = []
        accuracies = []
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_features, batch_labels in self.train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in self.test_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    test_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            test_loss /= len(self.test_loader)
            test_losses.append(test_loss)
            accuracy = 100 * correct / total
            accuracies.append(accuracy)
            
            print(f'Epoch [{epoch+1}/{self.num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Test Loss: {test_loss:.4f}, '
                  f'Accuracy: {accuracy:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(test_loss)
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_loss = test_loss
                print(f"New best model! Accuracy: {accuracy:.2f}%")
                self.save_model(self.model.state_dict(), accuracy)
        
        # Plot training curves
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Test Losses')
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Test Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.user_data_dir, 'training_curves.png'))
        plt.close()
        
        print(f"\nTraining completed!")
        print(f"Best accuracy: {best_accuracy:.2f}%")
        print(f"Best test loss: {best_loss:.4f}")
        
    def load_trained_model(self):
        """Load the trained model"""
        try:
            # Load model checkpoint with weights_only=False since we need to load the LabelEncoder
            checkpoint = torch.load(
                self.model_path,
                weights_only=False,
                map_location=self.device
            )
            
            # Load label encoder and number of classes
            self.label_encoder = checkpoint['label_encoder']
            self.num_classes = checkpoint['num_classes']
            
            # Create and load model
            self.model = GestureNet(self.input_size, self.hidden_size, 
                                  self.num_classes).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"\nLoaded model with {self.num_classes} classes")
            print(f"Best accuracy: {checkpoint['accuracy']:.2f}%")
            
        except Exception as e:
            print(f"\nError loading model: {e}")
            print("\nTrying alternative loading method...")
            try:
                # Try loading just the model weights if the full checkpoint fails
                self.model = GestureNet(self.input_size, self.hidden_size, 
                                      self.num_classes).to(self.device)
                self.model.load_state_dict(torch.load(
                    self.model_path,
                    map_location=self.device
                ))
                self.model.eval()
                print("Model weights loaded successfully")
                
                # Try to load label mapping separately
                if os.path.exists(self.label_mapping_file):
                    with open(self.label_mapping_file, 'r') as f:
                        label_mapping = json.load(f)
                    print("Label mapping loaded successfully")
                    
            except Exception as e2:
                print(f"\nError in alternative loading method: {e2}")
                print("\nPlease retrain the model:")
                print("1. Delete the contents of the gesture_data directory")
                print("2. Run: python main.py --collect-data")
                print("3. Run: python main.py --train-model")
                raise

    def save_model(self, state_dict, accuracy):
        """Save model with proper serialization"""
        save_dict = {
            'model_state_dict': state_dict,
            'label_encoder': self.label_encoder,
            'num_classes': self.num_classes,
            'accuracy': accuracy
        }
        
        # Save with proper serialization settings
        torch.save(save_dict, self.model_path)
        
        # Also save label mapping separately for backup
        label_mapping = {
            str(label): int(encoded_label)
            for label, encoded_label in zip(
                self.label_encoder.classes_,
                self.label_encoder.transform(self.label_encoder.classes_)
            )
        }
        with open(self.label_mapping_file, 'w') as f:
            json.dump(label_mapping, f, indent=2)

    def predict(self, features):
        """Predict gesture from features"""
        try:
            with torch.no_grad():
                # Reshape features to match LSTM input requirements
                # LSTM expects input of shape (batch_size, sequence_length, input_size)
                features = torch.FloatTensor(features)
                
                # If features is 4D (batch, frames, landmarks, coords), reshape to 3D
                if len(features.shape) == 4:
                    batch_size, frames, landmarks, coords = features.shape
                    features = features.reshape(batch_size, frames, landmarks * coords)
                
                # If features is 3D but in wrong order, transpose it
                elif len(features.shape) == 3:
                    # features = features.transpose(1, 2).contiguous()
                    pass
                
                # Move to device
                features = features.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                _, predicted = torch.max(outputs.data, 1)
                
                # Convert back to original label
                return self.label_encoder.inverse_transform([predicted.item()])[0]
                
        except Exception as e:
            print(f"\nError in prediction: {e}")
            print("Feature shape details:")
            print(f"Original shape: {features.shape}")
            print("Expected shape: (batch_size, sequence_length, input_features)")
            print("Where input_features should be 63 (21 landmarks * 3 coordinates)")
            raise

if __name__ == "__main__":
    trainer = GestureModelTrainer()
    trainer.train_model() 