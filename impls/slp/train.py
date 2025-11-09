"| Single Layer Perceptron - From Scratch Implementation |"

import jax
import wandb
from jax import random
import jax.numpy as jnp
from typing import Tuple

"""
 Activation Functions 
"""

def step_function(x: jnp.ndarray) -> jnp.ndarray:
    """
    Step activation function (Heaviside step function) - The ONLY activation for classic perceptron.
    
    Mathematical Representation:
    ----------------------------
    f(x) = { 1 if x > 0
            { 0 if x <= 0
    
    Note: Not differentiable at x=0, so we can't use gradient descent!
    
    
    WHY STEP FUNCTION ONLY? (Why not sigmoid even though it "works"?)
    ===================================================================
    
    1. Historical Accuracy
    ----------------------
    This is the ORIGINAL perceptron activation (Rosenblatt, 1958).
    The classic perceptron algorithm was designed for binary decisions.
    
    2. Algorithm Clarity
    --------------------
    - Perceptron Learning Rule is designed for discrete outputs (0 or 1)
    - Step function → clear right/wrong predictions → simple updates
    - Sigmoid → continuous outputs (0.0 to 1.0) → different algorithm!
    
    3. Conceptual Correctness (MOST IMPORTANT!)
    -------------------------------------------
    Mixing sigmoid with perceptron learning rule is CONCEPTUALLY WRONG:
    
    - With step function:
      → True Perceptron Learning Rule
      → Heuristic update based on binary errors
      → Perceptron Convergence Theorem applies
    
    - With sigmoid:
      → Actually "Delta Rule" (Widrow-Hoff, 1960)
      → Gradient descent on MSE loss
      → Different algorithm with different theory!
    
    Using sigmoid here would be teaching TWO algorithms at once
    while pretending it's one. Bad pedagogy!
    
    4. The Limitation is the Point!
    --------------------------------
    Step function's non-differentiability is a FEATURE for learning:
    
    - Shows why we can't use gradient descent
    - Motivates the need for continuous activations
    - Explains why perceptron is limited
    - Sets up the "why" for everything that comes next:
      * Sigmoid, tanh, ReLU (continuous & differentiable)
      * Gradient descent & backpropagation
      * Multi-layer networks (MLP)
      * Modern deep learning
    
    
    Where You'll Use Sigmoid (and why it makes sense there):
    ========================================================
    
    In the MLP implementation (next!):
    - With GRADIENT DESCENT (not perceptron rule)
    - With BACKPROPAGATION through layers
    - Where DIFFERENTIABILITY is required
    - In the PROPER theoretical context
    
    
    Bottom Line:
    ============
    This implementation is intentionally "limited" to teach the
    classic perceptron correctly. Learn one algorithm well, then
    move to the next. The limitations you discover here are not
    bugs - they're the motivation for 60+ years of research!
    
    
    Reference:
    ==========
    Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model
    for Information Storage and Organization in the Brain"
    """
    return jnp.where(x > 0, 1.0, 0.0)


"""
 Single Layer Perceptron Class
"""

class SingleLayerPerceptron:
    """ Single Layer Perceptron for binary classification. """
    
    def __init__(
        self, 
        n_features: int, 
        learning_rate: float = 0.1,
        random_seed: int = 42
    ):
        """
        Initialize the Single Layer Perceptron.
        """
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.random_seed = random_seed

        # Activation function is ALWAYS step function (classic perceptron)
        self.activation = step_function
        
        # =====================================================================
        # Weight Initialization: Xavier/Glorot Initialization method
        # =====================================================================
        # 
        # What is Xavier Initialization?
        # ------------------------------
        # Xavier (or Glorot) initialization is a weight initialization technique
        # proposed by Xavier Glorot and Yoshua Bengio in 2010. It helps neural
        # networks train faster and reach better performance by setting initial
        # weights to appropriate values.
        #
        # The Problem it Solves:
        # ----------------------
        # - If weights are too small: signals shrink as they pass through layers
        #   (vanishing gradients)
        # - If weights are too large: signals explode as they pass through layers
        #   (exploding gradients)
        # - Xavier initialization finds a "sweet spot" to keep signal variance
        #   roughly constant across layers
        #
        # The Formula:
        # ------------
        # For a layer with n_in inputs and n_out outputs:
        #   
        #   Standard Xavier:  scale = sqrt(2 / (n_in + n_out))
        #   Xavier Uniform:   W ~ Uniform(-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out)))
        #   Xavier Normal:    W ~ Normal(0, sqrt(2/(n_in+n_out)))
        #
        # Our Scenario (Single Layer Perceptron):
        # ----------------------------------------
        # - n_in = n_features (input dimension, e.g., 2 for 2D data)
        # - n_out = 1 (output dimension, binary classification)
        # - We use a simplified version: scale = sqrt(2 / n_features)
        #
        # Example with our 2D dataset:
        # - n_features = 2
        # - scale = sqrt(2 / 2) = sqrt(1) = 1.0
        # - weights ~ Normal(0, 1.0)
        # - This keeps the variance of activations around 1.0
        #
        # Why it matters for perceptron:
        # - With good initialization, the perceptron converges faster
        # - Prevents extreme initial predictions that confuse learning
        # - Especially important when scaling to higher dimensions
        
        # Generate random keys for reproducibility
        key = random.PRNGKey(random_seed)
        key_w, key_b = random.split(key)  # Separate keys for weights and bias
        
        # Compute Xavier scaling factor
        scale = jnp.sqrt(2.0 / n_features)
        
        # Initialize weights from Normal(0, scale)
        # Shape: (n_features,) - one weight per input feature
        self.weights = random.normal(key_w, (n_features,)) * scale
        
        # Initialize bias from Normal(0, scale)
        # Shape: scalar - single bias for the output
        self.bias = random.normal(key_b, ()) * scale
        
        # Track training history
        self.history = {
            'loss': [],
            'accuracy': []
        }

    def forward(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass: compute predictions.
        
        Mathematical Representation:
        y = activation(w · x + b)
        """
        # Linear combination: z = w · x + b
        z = jnp.dot(X, self.weights) + self.bias
        
        # Apply activation function
        return self.activation(z)
    
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Make binary predictions.
        
        Mathematical Representation:
        y = activation(z)
        """
        return self.forward(X)
    
    def update_weights(
        self, 
        X: jnp.ndarray, 
        y_true: jnp.ndarray, 
        y_pred: jnp.ndarray
    ) -> int:
        """
        Update weights using the Single Layer Perceptron learning rule.
        
        Mathematical Representation:
        w_new = w_old + learning_rate * (y_true - y_pred) * x
        b_new = b_old + learning_rate * (y_true - y_pred)
        """
        # =====================================================================
        # Error Calculation: Simple Difference (NOT MSE!)
        # =====================================================================
        # 
        # Why simple difference instead of MSE here?
        # -------------------------------------------
        # This is the PERCEPTRON LEARNING RULE, not gradient descent!
        # 
        # Perceptron Learning Rule uses:
        #   error = y_true - y_pred
        # 
        # NOT:
        #   error = (y_true - y_pred)²  (this would be MSE)
        # 
        # Key Difference:
        # ---------------
        # - Perceptron algorithm: Uses raw error signal directly
        # - Gradient-based methods: Use loss function derivatives (like MSE gradient)
        # 
        # Why this works for perceptron:
        # -------------------------------
        # - For binary classification (0 or 1):
        #   * Correct prediction: error = 0 (no update)
        #   * Wrong prediction: error = ±1 (directional update)
        # - The sign and magnitude of error tells us how to adjust weights
        # - Perceptron Convergence Theorem guarantees this converges
        #   for linearly separable data
        # 
        # Example:
        # --------
        # True = 1, Pred = 0 → error = +1 → increase weights
        # True = 0, Pred = 1 → error = -1 → decrease weights
        # True = 1, Pred = 1 → error = 0  → no change
        # 
        # If we used MSE:
        # ---------------
        # MSE = (y_true - y_pred)²
        # - All errors would be positive (squared)
        # - We'd lose directional information (should we increase or decrease?)
        # - Would need to compute gradient: ∂MSE/∂w = -2(y - ŷ)x
        # =====================================================================
        
        errors = y_true - y_pred
        
        # =====================================================================
        # Weight Update Rule (Vectorized Implementation)
        # =====================================================================
        # 
        # Mathematical Form (for single sample):
        # --------------------------------------
        # w_new = w_old + learning_rate * error * x
        # 
        # where:
        #   - error = (y_true - y_pred)
        #   - x is the input feature vector
        # 
        # Intuition:
        # ----------
        # - If we predict 0 but true is 1 (error = +1): increase weights in direction of x
        # - If we predict 1 but true is 0 (error = -1): decrease weights in direction of x
        # - If prediction is correct (error = 0): no change
        # 
        # Vectorized Form (for batch of samples):
        # ----------------------------------------
        # weight_update = learning_rate * (1/n) * Σ(error_i * x_i)
        #               = learning_rate * (1/n) * X.T @ errors
        # 
        # Breaking down jnp.dot(X.T, errors):
        # ------------------------------------
        # X shape:      (n_samples, n_features)  e.g., (100, 2)
        # X.T shape:    (n_features, n_samples)  e.g., (2, 100)
        # errors shape: (n_samples,)             e.g., (100,)
        # 
        # X.T @ errors = [w1_update, w2_update, ...]  shape: (n_features,)
        # 
        # Example with n_features=2:
        # X.T @ errors = [x1_1*e_1 + x1_2*e_2 + ... + x1_n*e_n,
        #                 x2_1*e_1 + x2_2*e_2 + ... + x2_n*e_n]
        # 
        # Each weight gets updated based on how much its corresponding
        # feature contributed to the errors across all samples.
        # 
        # Why divide by len(y_true)?
        # ---------------------------
        # - Averages the update across all samples
        # - Makes learning rate independent of batch size
        # - Ensures consistent learning behavior regardless of dataset size
        # =====================================================================

        weight_update = self.learning_rate * jnp.dot(X.T, errors) / len(y_true)
        
        # =====================================================================
        # Bias Update Rule
        # =====================================================================
        # 
        # Mathematical Form:
        # ------------------
        # b_new = b_old + learning_rate * mean(errors)
        # 
        # Why use mean?
        # -------------
        # - Bias doesn't have an input feature to multiply with
        # - We average the errors across all samples
        # - This shifts the decision boundary up/down to reduce overall error
        # 
        # Example:
        # --------
        # - If most predictions are too low (errors > 0): increase bias
        # - If most predictions are too high (errors < 0): decrease bias
        # =====================================================================
        
        bias_update = self.learning_rate * jnp.mean(errors)
        
        # Update weights and bias using calculated updates
        self.weights = self.weights + weight_update
        self.bias = self.bias + bias_update
        
        # Count misclassifications: number of predictions that are not 0 or 1
        n_errors = jnp.sum(jnp.abs(errors) > 0.5)
        return int(n_errors)
    
    def fit(
        self, 
        X: jnp.ndarray, 
        y: jnp.ndarray, 
        epochs: int = 100,
        verbose: bool = True,
        log_wandb: bool = True
    ):
        """
        Train the Single Layer Perceptron using classic perceptron learning rule.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels (0 or 1), shape (n_samples,)
            epochs: Maximum number of training epochs
            verbose: Print training progress
            log_wandb: Log metrics to Weights & Biases
        """
        n_samples = len(y)
        
        for epoch in range(epochs):
            # Forward pass: compute predictions
            predictions = self.forward(X)
            
            # Update weights: adjust weights and bias based on errors
            n_errors = self.update_weights(X, y, predictions)
            
            # Calculate metrics: accuracy and loss
            accuracy = 1.0 - (n_errors / n_samples)
            
            # =====================================================================
            # MSE Loss Calculation (For Monitoring ONLY, not for training!)
            # =====================================================================
            # 
            # Why calculate MSE here when we don't use it for updates?
            # ---------------------------------------------------------
            # MSE is calculated ONLY for monitoring/logging purposes!
            # 
            # Two Different Purposes:
            # -----------------------
            # 1. Training Algorithm (weight updates):
            #    - Uses: error = y_true - y_pred
            #    - Purpose: Perceptron learning rule
            #    - Directly updates weights
            # 
            # 2. Progress Monitoring (this MSE):
            #    - Uses: loss = mean((y_true - y_pred)²)
            #    - Purpose: Track training progress
            #    - NOT used for weight updates
            # 
            # Why MSE for monitoring?
            # -----------------------
            # - Gives us a single number to track improvement
            # - Always positive (easy to visualize decrease)
            # - Smooth metric (unlike accuracy which is discrete)
            # - Standard metric for comparison with other algorithms
            # - Helps us see convergence even when accuracy is already 100%
            # 
            # Example:
            # --------
            # Epoch 1: Accuracy 85%, Loss 0.15 → Model struggling
            # Epoch 5: Accuracy 95%, Loss 0.05 → Getting better
            # Epoch 10: Accuracy 100%, Loss 0.00 → Perfect!
            # 
            # Important Distinction:
            # ----------------------
            # - Perceptron DOES NOT minimize MSE during training
            # - It follows the perceptron convergence algorithm
            # - MSE just helps us understand training progress
            # - In MLP/modern networks, we WOULD minimize MSE via gradient descent
            # =====================================================================
            
            loss = jnp.mean((y - predictions) ** 2)  # MSE for monitoring only!
            
            # Store history: store loss and accuracy for each epoch
            self.history['loss'].append(float(loss))
            self.history['accuracy'].append(float(accuracy))
            
            # Log to Weights & Biases
            if log_wandb:
                wandb.log({
                    "loss": float(loss),
                    "accuracy": float(accuracy),
                    "errors": int(n_errors),
                    "epoch": epoch + 1
                })
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1:3d}/{epochs} | "
                      f"Loss: {loss:.4f} | "
                      f"Acc: {accuracy:.4f} | "
                      f"Errors: {n_errors}/{n_samples}")
            
            # Early stopping if perfect classification
            if n_errors == 0 and verbose:
                print(f"  Converged at epoch {epoch + 1}")
                break
    
    def get_decision_boundary(self) -> Tuple[float, float]:
        """
        Get the decision boundary for 2D data.
        
        Mathematical Representation:
        w1*x1 + w2*x2 + b = 0
        Solving for x2: x2 = -(w1*x1 + b) / w2
        """
        if self.n_features != 2:
            raise ValueError("Decision boundary only defined for 2D data")
        
        w1, w2 = self.weights
        slope = -w1 / w2
        intercept = -self.bias / w2
        return slope, intercept


# =============================================================================
# Data Generation
# =============================================================================

def generate_linearly_separable_data(
    n_samples: int = 100, 
    random_seed: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a simple linearly separable dataset for binary classification.
    """
    key = random.PRNGKey(random_seed)
    key1, key2 = random.split(key)
    
    # Class 0: centered around (-1, -1)
    n_class0 = n_samples // 2
    X_class0 = random.normal(key1, (n_class0, 2)) * 0.5 + jnp.array([-1.5, -1.5])
    y_class0 = jnp.zeros(n_class0)
    
    # Class 1: centered around (1, 1)
    n_class1 = n_samples - n_class0
    X_class1 = random.normal(key2, (n_class1, 2)) * 0.5 + jnp.array([1.5, 1.5])
    y_class1 = jnp.ones(n_class1)
    
    # Combine and shuffle
    X = jnp.vstack([X_class0, X_class1])
    y = jnp.concatenate([y_class0, y_class1])
    
    # Shuffle
    key_shuffle = random.PRNGKey(random_seed + 1)
    perm = random.permutation(key_shuffle, n_samples)
    X = X[perm]
    y = y[perm]
    
    return X, y


# =============================================================================
# Logging Helpers
# =============================================================================

def log_data_to_wandb(X: jnp.ndarray, y: jnp.ndarray, prefix: str = "data"):
    """Log dataset as scatter plot to wandb."""
    data = []
    for i in range(len(X)):
        data.append([float(X[i, 0]), float(X[i, 1]), int(y[i])])
    
    table = wandb.Table(data=data, columns=["feature_1", "feature_2", "class"])
    wandb.log({
        f"{prefix}/scatter": wandb.plot.scatter(
            table, "feature_1", "feature_2", "class",
            title=f"{prefix.capitalize()} Distribution"
        )
    })

def log_decision_boundary(perceptron: SingleLayerPerceptron, X: jnp.ndarray, y: jnp.ndarray, prefix: str):
    """Log decision boundary data to wandb."""
    # Get decision boundary line
    x1_min, x1_max = float(X[:, 0].min() - 1), float(X[:, 0].max() + 1)
    slope, intercept = perceptron.get_decision_boundary()
    
    # Create points for decision boundary
    x1_points = jnp.linspace(x1_min, x1_max, 50)
    x2_points = slope * x1_points + intercept
    
    # Log data points with classes
    data_points = []
    for i in range(len(X)):
        data_points.append([
            float(X[i, 0]), 
            float(X[i, 1]), 
            "Class 0" if y[i] == 0 else "Class 1"
        ])
    
    # Log boundary points
    for i in range(len(x1_points)):
        data_points.append([
            float(x1_points[i]), 
            float(x2_points[i]), 
            "Decision Boundary"
        ])
    
    table = wandb.Table(
        data=data_points, 
        columns=["feature_1", "feature_2", "type"]
    )
    wandb.log({
        f"{prefix}/decision_boundary": wandb.plot.scatter(
            table, "feature_1", "feature_2", title=f"{prefix.capitalize()} Decision Boundary"
        )
    })


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    """Main training and evaluation pipeline."""
    print("\nSINGLE LAYER PERCEPTRON")
    print("_" * 70)
    
    # Initialize Weights & Biases
    print("\nInitializing Weights & Biases...")
    wandb.init(
        project="slp-from-scratch",
        name="classic-perceptron",
        config={
            "activation": "step",
            "n_samples": 100,
            "epochs": 100,
            "learning_rate": 0.1,
            "random_seed": 42
        },
        tags=["perceptron", "binary-classification", "jax", "rosenblatt-1958"]
    )
    
    # Generate data
    print("\nGenerating dataset...")
    X, y = generate_linearly_separable_data(n_samples=100, random_seed=42)
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Class 0: {jnp.sum(y == 0)} | Class 1: {jnp.sum(y == 1)}")
    
    # Train classic perceptron
    print("\nTraining Classic Perceptron")
    print("_" * 70)
    perceptron = SingleLayerPerceptron(
        n_features=2, 
        learning_rate=0.1,
        random_seed=42
    )
    perceptron.fit(X, y, epochs=100, verbose=True)
    
    # Evaluate
    print("\nEvaluation")
    predictions = perceptron.predict(X)
    accuracy = jnp.mean(predictions == y)
    print(f"  Final Accuracy: {accuracy:.4f}")
    print(f"  Learned Weights: {perceptron.weights}")
    print(f"  Learned Bias: {perceptron.bias:.4f}")
    
    # Log final metrics to wandb
    wandb.log({
        "final_accuracy": float(accuracy),
        "weight_norm": float(jnp.linalg.norm(perceptron.weights)),
        "bias": float(perceptron.bias)
    })
    
    # Log decision boundary to W&B
    print("\nUploading decision boundary to W&B...")
    log_decision_boundary(perceptron, X, y, "perceptron")
    
    print(f"\nDashboard: {wandb.run.get_url()}")
    print("  - Interactive charts auto-generated from metrics")
    print("  - Decision boundaries visualized")
    print("  - Training curves available in panels")
    
    wandb.finish()
    
    print("\nTraining complete")
    print("_" * 70 + "\n")


if __name__ == "__main__":
    main()