import torch
import torch.nn as nn

# Simple classifier class
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=16, device=None):
        super(SimpleClassifier, self).__init__()

        # Define the classifier structure
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_units),  # First linear layer with hidden units
            nn.ReLU(),  # Activation function
            nn.Linear(hidden_units, output_size),  # Final linear layer mapping to output
            nn.Sigmoid()  # Sigmoid to map output to range [0, 1]
        )

        # Apply initialization function and move to device (if specified)
        self.apply(self.weights_init)
        if device:
            self.to(device)

    # Custom weight initialization
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, x):
        # Pass the input through the classifier
        out = self.classifier(x)
        return out  # Output should be a single value (for binary classification)
