"""| Multi-Layer Perceptron - From Scratch Implementation |"""

import jax
import wandb
from jax import random
import jax.numpy as jnp
from typing import Tuple, List

"""
 Activation Functions 
"""

def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """
    Sigmoid activation function.
    
    Mathematical Representation:
    ----------------------------
    f(x) = 1 / (1 + e^(-x))
    
    Properties:
    -----------
    - Output range: (0, 1)
    - Smooth and differentiable everywhere
    - Used in hidden layers and output layer for binary classification
    
    Why sigmoid for MLP (vs step for perceptron)?
    ==============================================
    - Differentiable: Required for gradient descent and backpropagation
    - Continuous: Allows gradual weight updates, not just binary jumps
    - Enables multi-layer learning: Gradients can flow backwards through layers
    """
    return 1 / (1 + jnp.exp(-x))


def sigmoid_derivative(a: jnp.ndarray) -> jnp.ndarray:
    """
    Derivative of sigmoid with respect to its output.
    
    Mathematical Representation:
    ----------------------------
    f'(x) = f(x) * (1 - f(x))
    
    Note: We use 'a' (activation output) instead of 'x' (pre-activation)
    because: f'(x) = a * (1 - a) where a = f(x)
    
    This is more efficient as we already computed a = f(x) in forward pass!
    """
    return a * (1 - a)


"""
 Multi-Layer Perceptron Class
"""

class MultiLayerPerceptron:
    """Multi-Layer Perceptron with manual backpropagation implementation."""
    
    def __init__(
        self, 
        layer_sizes: List[int],
        learning_rate: float = 1.0,
        random_seed: int = 42
    ):
        """
        Initialize the Multi-Layer Perceptron.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
                        Example: [2, 4, 4, 1] means:
                        - 2 input features
                        - 4 neurons in first hidden layer
                        - 4 neurons in second hidden layer
                        - 1 output neuron
            learning_rate: Learning rate for gradient descent
            random_seed: Random seed for reproducibility
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.n_layers = len(layer_sizes) - 1  # Number of weight matrices
        
        # Initialize parameters
        self.params = self._init_params()
        
        # Track training history
        self.history = {
            'loss': [],
            'accuracy': []
        }
    
    def _init_params(self) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Initialize network parameters using Xavier/Glorot initialization.
        
        Xavier Initialization for Deep Networks:
        ----------------------------------------
        - For layer with n_in inputs and n_out outputs:
          scale = sqrt(2 / (n_in + n_out))
        - Simplified version: scale = 0.5 (works well for small networks)
        - Helps maintain stable gradients during backpropagation
        
        Returns:
            List of (W, b) tuples for each layer
        """
        key = random.PRNGKey(self.random_seed)
        keys = random.split(key, self.n_layers)
        
        params = []
        for i, k in enumerate(keys):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            
            # Initialize weights with small random values
            W = random.normal(k, (n_in, n_out)) * 0.5
            
            # Initialize biases to zero
            b = jnp.zeros((1, n_out))
            
            params.append((W, b))
        
        return params
    
    def forward(self, X: jnp.ndarray) -> Tuple:
        """
        Forward pass through the network.
        
        Returns all intermediate activations for backpropagation.
        
        Mathematical Flow (for 3-layer network):
        ----------------------------------------
        Layer 1: z1 = X @ W1 + b1  →  a1 = σ(z1)
        Layer 2: z2 = a1 @ W2 + b2  →  a2 = σ(z2)
        Layer 3: z3 = a2 @ W3 + b3  →  a3 = σ(z3)
        
        Returns:
            Tuple of all z (pre-activations) and a (activations)
        """
        activations = []
        a = X
        
        # Forward through all layers
        for W, b in self.params:
            z = a @ W + b
            a = sigmoid(z)
            activations.append((z, a))
        
        return activations
    
    def _compute_loss(self, y_pred: jnp.ndarray, y_true: jnp.ndarray) -> float:
        """
        Compute Mean Squared Error loss.
        
        Mathematical Representation:
        ----------------------------
        MSE = (1/m) * Σ(y_true - y_pred)²
        
        Why MSE for MLP?
        ----------------
        - Differentiable: Can compute gradients for backpropagation
        - Convex for linear models: Guarantees convergence to global minimum
        - Penalizes large errors more: Squared term emphasizes bigger mistakes
        - Standard for regression and binary classification with sigmoid output
        """
        return jnp.mean((y_pred - y_true) ** 2)
    
    def backward(
        self, 
        X: jnp.ndarray, 
        y_true: jnp.ndarray, 
        activations: List[Tuple[jnp.ndarray, jnp.ndarray]]
    ) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Backward pass: compute gradients using manual backpropagation.
        
        Backpropagation Algorithm:
        --------------------------
        1. Compute output layer error: dL/da_out
        2. Backpropagate error through layers using chain rule
        3. Compute weight and bias gradients at each layer
        
        Chain Rule for MLP (3-layer example):
        --------------------------------------
        Output layer (L3):
            dL/dW3 = dL/da3 * da3/dz3 * dz3/dW3
                   = (a3 - y) * σ'(z3) * a2.T
        
        Hidden layer (L2):
            dL/dW2 = dL/da3 * da3/dz3 * dz3/da2 * da2/dz2 * dz2/dW2
                   = (dL/dz3 @ W3.T) * σ'(z2) * a1.T
        
        Key Insight:
        ------------
        Errors propagate backwards, gradients computed layer by layer.
        Each layer's error = next layer's error * weight * activation derivative
        """
        m = X.shape[0]  # batch size
        grads = []
        
        # Get final output
        y_pred = activations[-1][1]
        
        # ====================================================================
        # Output Layer Gradient
        # ====================================================================
        # dL/da = (a - y)  [derivative of MSE w.r.t. output]
        # dL/dz = dL/da * σ'(a)  [chain rule]
        dA = y_pred - y_true
        dZ = dA * sigmoid_derivative(y_pred)
        
        # Backpropagate through all layers (reverse order)
        for i in range(self.n_layers - 1, -1, -1):
            # Get current layer activations
            z_curr, a_curr = activations[i]
            
            # Get previous layer activation (or input X for first layer)
            if i == 0:
                a_prev = X
            else:
                a_prev = activations[i - 1][1]
            
            # Compute weight and bias gradients
            dW = (a_prev.T @ dZ) / m
            db = jnp.sum(dZ, axis=0, keepdims=True) / m
            
            # Store gradients (will reverse later)
            grads.append((dW, db))
            
            # Backpropagate error to previous layer (if not input layer)
            if i > 0:
                W_curr = self.params[i][0]
                dA_prev = dZ @ W_curr.T
                dZ = dA_prev * sigmoid_derivative(activations[i - 1][1])
        
        # Reverse gradients to match forward order
        grads.reverse()
        
        return grads
    
    def update_weights(self, grads: List[Tuple[jnp.ndarray, jnp.ndarray]]):
        """
        Update network parameters using gradient descent.
        
        Mathematical Representation:
        ----------------------------
        W_new = W_old - learning_rate * dL/dW
        b_new = b_old - learning_rate * dL/db
        
        Why subtract gradients?
        -----------------------
        - Gradient points in direction of steepest increase in loss
        - We want to decrease loss, so we move opposite to gradient
        - Learning rate controls step size
        """
        new_params = []
        for (W, b), (dW, db) in zip(self.params, grads):
            W_new = W - self.learning_rate * dW
            b_new = b - self.learning_rate * db
            new_params.append((W_new, b_new))
        
        self.params = new_params
    
    def fit(
        self, 
        X: jnp.ndarray, 
        y: jnp.ndarray, 
        epochs: int = 10000,
        verbose: bool = True,
        log_wandb: bool = True,
        log_interval: int = 1000
    ):
        """
        Train the Multi-Layer Perceptron using gradient descent.
        
        Training Loop:
        --------------
        1. Forward pass: compute predictions and activations
        2. Compute loss
        3. Backward pass: compute gradients
        4. Update weights using gradient descent
        5. Log metrics and repeat
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples, 1)
            epochs: Number of training epochs
            verbose: Print training progress
            log_wandb: Log metrics to Weights & Biases
            log_interval: Print/log every N epochs
        """
        n_samples = len(y)
        
        for epoch in range(epochs):
            # Forward pass
            activations = self.forward(X)
            y_pred = activations[-1][1]
            
            # Compute loss
            loss = self._compute_loss(y_pred, y)
            
            # Backward pass
            grads = self.backward(X, y, activations)
            
            # Update weights
            self.update_weights(grads)
            
            # Compute accuracy (for binary classification)
            predictions_binary = (y_pred > 0.5).astype(jnp.float32)
            accuracy = jnp.mean(predictions_binary == y)
            
            # Store history
            self.history['loss'].append(float(loss))
            self.history['accuracy'].append(float(accuracy))
            
            # Log to Weights & Biases
            if log_wandb:
                wandb.log({
                    "loss": float(loss),
                    "accuracy": float(accuracy),
                    "epoch": epoch + 1
                })
            
            # Print progress
            if verbose and (epoch % log_interval == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch:5d}/{epochs} | "
                      f"Loss: {loss:.6f} | "
                      f"Acc: {accuracy:.4f}")
    
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Make predictions on new data.
        
        Returns:
            Predictions (probabilities between 0 and 1)
        """
        activations = self.forward(X)
        return activations[-1][1]


# =============================================================================
# Data Generation
# =============================================================================

def generate_xor_data() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate XOR dataset.
    
    XOR Problem:
    ------------
    The classic test for multi-layer networks!
    - Single layer perceptron CANNOT solve XOR (not linearly separable)
    - Multi-layer perceptron CAN solve XOR (learns non-linear decision boundary)
    
    Truth Table:
    ------------
    Input1  Input2  Output
      0       0       0
      0       1       1
      1       0       1
      1       1       0
    """
    X = jnp.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ])
    y = jnp.array([[0.], [1.], [1.], [0.]])
    
    return X, y


# =============================================================================
# Logging Helpers
# =============================================================================

def log_predictions_to_wandb(mlp: MultiLayerPerceptron, X: jnp.ndarray, y: jnp.ndarray):
    """Log final predictions as a table to wandb."""
    predictions = mlp.predict(X)
    
    data = []
    for i in range(len(X)):
        data.append([
            float(X[i, 0]),
            float(X[i, 1]),
            float(y[i, 0]),
            float(predictions[i, 0]),
            int(predictions[i, 0] > 0.5)
        ])
    
    table = wandb.Table(
        data=data,
        columns=["input_1", "input_2", "true_label", "prediction", "predicted_class"]
    )
    wandb.log({"predictions": table})


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    """Main training and evaluation pipeline."""
    print("\nMULTI-LAYER PERCEPTRON (MLP)")
    print("_" * 70)
    
    # Initialize Weights & Biases
    print("\nInitializing Weights & Biases...")
    wandb.init(
        project="mlp-from-scratch",
        name="xor-problem",
        config={
            "architecture": [2, 4, 4, 1],
            "activation": "sigmoid",
            "loss": "mse",
            "optimizer": "gradient_descent",
            "learning_rate": 1.0,
            "epochs": 10000,
            "random_seed": 0
        },
        tags=["mlp", "xor", "backpropagation", "jax", "from-scratch"]
    )
    
    # Generate XOR data
    print("\nGenerating XOR dataset...")
    X, y = generate_xor_data()
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"\nXOR Truth Table:")
    print("  Input1  Input2  →  Output")
    for i in range(len(X)):
        print(f"    {int(X[i, 0])}       {int(X[i, 1])}     →    {int(y[i, 0])}")
    
    # Create and train MLP
    print("\nTraining Multi-Layer Perceptron")
    print("_" * 70)
    mlp = MultiLayerPerceptron(
        layer_sizes=[2, 4, 4, 1],
        learning_rate=1.0,
        random_seed=0
    )
    
    print(f"\nNetwork Architecture:")
    print(f"  Input layer:    {mlp.layer_sizes[0]} neurons")
    for i in range(1, len(mlp.layer_sizes) - 1):
        print(f"  Hidden layer {i}: {mlp.layer_sizes[i]} neurons")
    print(f"  Output layer:   {mlp.layer_sizes[-1]} neuron")
    print(f"\nTotal parameters: {sum(W.size + b.size for W, b in mlp.params)}")
    
    mlp.fit(X, y, epochs=10000, verbose=True, log_interval=1000)
    
    # Final evaluation
    print("\nEvaluation")
    print("_" * 70)
    predictions = mlp.predict(X)
    predictions_binary = (predictions > 0.5).astype(jnp.float32)
    accuracy = jnp.mean(predictions_binary == y)
    final_loss = mlp._compute_loss(predictions, y)
    
    print(f"\nFinal Results:")
    print(f"  Loss:     {final_loss:.6f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    print(f"\nPredictions (rounded to 3 decimals):")
    print("  Input      True    Predicted    Binary")
    for i in range(len(X)):
        print(f"  {X[i]}  →  {int(y[i, 0])}    {predictions[i, 0]:.3f}        {int(predictions_binary[i, 0])}")
    
    # Log final metrics to wandb
    wandb.log({
        "final_loss": float(final_loss),
        "final_accuracy": float(accuracy)
    })
    
    # Log predictions table
    log_predictions_to_wandb(mlp, X, y)
    
    print(f"\nDashboard: {wandb.run.get_url()}")
    print("  - Loss and accuracy curves available")
    print("  - Predictions table logged")
    print("  - Architecture and hyperparameters saved")
    
    wandb.finish()
    
    print("\nTraining complete!")
    print("_" * 70 + "\n")


if __name__ == "__main__":
    main()