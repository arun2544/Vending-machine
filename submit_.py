import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Global scaler to be reused between fit and decode if needed
scaler = None

################################
# Non Editable Region Starting #
################################
def my_fit(X_train, y_train):
################################
#  Non Editable Region Ending  #
################################
    global scaler
    X_mapped = my_map(X_train)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mapped)

    model = LogisticRegression(
        penalty='l2',
        C=3.3,
        solver='lbfgs',
        max_iter=2000,
        tol=1e-5,
        random_state=42
    )
    model.fit(X_scaled, y_train)

    # Return weight vector and bias
    w = model.coef_.flatten()
    b = model.intercept_[0]
    return w, b

################################
# Non Editable Region Starting #
################################
def my_map(X):
################################
#  Non Editable Region Ending  #
################################
    X = 1 - 2 * X  # Convert {0,1} to {-1,+1}
    N, D = X.shape

    # φ0: Forward cumulative (9D)
    phi0 = np.column_stack([np.ones(N)] + [np.cumprod(X[:, :i+1], axis=1)[:, -1] for i in range(D)])

    # φ1: Reverse cumulative (9D)
    phi1 = np.column_stack([np.ones(N)] + [np.cumprod(X[:, i:], axis=1)[:, 0] for i in range(D)])

    # Linear terms (8D)
    first_order = X

    # Quadratic terms (28D)
    second_order = np.column_stack([X[:, i] * X[:, j] for i in range(D) for j in range(i+1, D)])

    # Strategic third-order terms (12D)
    third_order = np.column_stack([
        X[:, 0] * X[:, 1] * X[:, 2],
        X[:, 1] * X[:, 2] * X[:, 3],
        X[:, 2] * X[:, 3] * X[:, 4],
        X[:, 3] * X[:, 4] * X[:, 5],
        X[:, 4] * X[:, 5] * X[:, 6],
        X[:, 5] * X[:, 6] * X[:, 7],
        X[:, 0] * X[:, 2] * X[:, 4],
        X[:, 1] * X[:, 3] * X[:, 5],
        X[:, 2] * X[:, 4] * X[:, 6],
        X[:, 3] * X[:, 5] * X[:, 7],
        X[:, 0] * X[:, 3] * X[:, 6],
        X[:, 1] * X[:, 4] * X[:, 7],
        X[:, 0] * X[:, 4] * X[:, 7],
        X[:, 1] * X[:, 5] * X[:, 6],
        X[:, 0] * X[:, 1] * X[:, 7],
        X[:, 2] * X[:, 3] * X[:, 7],
        X[:, 0] * X[:, 6] * X[:, 7],
        X[:, 1] * X[:, 5] * X[:, 7],
        X[:, 2] * X[:, 4] * X[:, 7]
    ])

    all_features = np.column_stack([phi0, phi1, first_order, second_order, third_order])
    return all_features[:, :64]  # 64 features (exclude bias here)

################################
# Non Editable Region Starting #
################################
def my_decode(w):
    """
    Recovers non-negative delays from 65D model using constrained least squares.
    Returns four 64-dim arrays (p, q, r, s) of non-negative delays.
    """
    # Extract weights and bias
    weights = w[:-1] if len(w) == 65 else w
    bias = w[-1] if len(w) == 65 else 0.0
    b_vec = np.append(weights, bias)
    
    # Construct coefficient matrix A (65x256)
    A = np.zeros((65, 256))
    for i in range(64):
        if i == 0:
            A[0, 0:4] = [0.5, -0.5, 0.5, -0.5]  # Stage 0
        else:
            A[i, 4*i:4*i+4] = [0.5, -0.5, 0.5, -0.5]  # Current stage
            A[i, 4*(i-1):4*(i-1)+4] = [0.5, -0.5, -0.5, 0.5]  # Previous stage
    A[64, 252:256] = [0.5, -0.5, -0.5, 0.5]  # Bias equation

    # Step 1: Initial unconstrained solution
    d = np.linalg.lstsq(A, b_vec, rcond=None)[0]
    
    # Step 2: Enforce non-negativity through iterative projection
    for _ in range(10):  # Limited iterations for efficiency
        # Project negative values to zero
        negative = d < 0
        if not np.any(negative):
            break
        d[negative] = 0
        
        # Re-solve for remaining active variables
        active = d > 0
        if np.any(active):
            d_active = np.linalg.lstsq(A[:, active], b_vec, rcond=None)[0]
            d[active] = np.maximum(0, d_active)  # Re-enforce non-negativity
    
    # Final cleanup and validation
    d = np.maximum(d, 0)  # Ensure strict non-negativity
    residual = np.linalg.norm(A @ d - b_vec)
    print(f"Reconstruction residual: {residual:.2e}")
    
    # Split into p, q, r, s components
    return d[0::4], d[1::4], d[2::4], d[3::4]  # Each 64-dimensional

def load_data(path):
    data = np.loadtxt(path)
    return data[:, :8], data[:, 8]

def evaluate(w, b, scaler, X_test, y_test):
    X_test_mapped = my_map(X_test)
    X_test_scaled = scaler.transform(X_test_mapped)
    preds = (X_test_scaled @ w + b >= 0).astype(int)
    acc = np.mean(preds == y_test)
    return acc

def check_model(p, q, r, s, target_weights, target_bias):
    """Helper function to verify reconstructed model"""
    A = np.zeros((65, 256))
    for i in range(64):
        if i == 0:
            A[0, 0:4] = [0.5, -0.5, 0.5, -0.5]
        else:
            A[i, 4*i:4*i+4] = [0.5, -0.5, 0.5, -0.5]
            A[i, 4*(i-1):4*(i-1)+4] = [0.5, -0.5, -0.5, 0.5]
    A[64, 252:256] = [0.5, -0.5, -0.5, 0.5]
    
    d = np.zeros(256)
    d[0::4] = p
    d[1::4] = q
    d[2::4] = r
    d[3::4] = s
    
    reconstructed = A @ d
    return (np.allclose(reconstructed[:-1], target_weights, atol=1e-4) and 
            np.isclose(reconstructed[-1], target_bias, atol=1e-4))

def main():
    # Part 1: Original PUF evaluation
    print("=== Standard PUF Evaluation ===")
    X_train, y_train = load_data("secret_trn.txt")
    X_test, y_test = load_data("secret_tst.txt")
    
    X_test_mapped = my_map(X_test)
    w, b = my_fit(X_train, y_train)
    acc = evaluate(w, b, scaler, X_test, y_test)
    
    print(f"\nTest Accuracy = {acc*100:.2f}%")
    print(f"Feature Dimension: {X_test_mapped.shape[1]}")
    
    # Part 2: Process public_mod.txt models
    print("\n\n=== Processing public_mod.txt ===")
    try:
        models = np.loadtxt("public_mod.txt")
        if models.ndim == 1:
            models = models.reshape(1, -1)  # Handle single model case
            
        print(f"Found {models.shape[0]} models in file")
        
        for i, model in enumerate(models):
            if len(model) != 65:
                print(f"Model {i+1} has invalid dimension {len(model)} (expected 65)")
                continue
                
            print(f"\nProcessing Model {i+1}:")
            p, q, r, s = my_decode(model)
            
            print("\ndelays of each type:")
            print(f"p: {p[:]}...")
            print(f"q: {q[:]}...")
            print(f"r: {r[:]}...")
            print(f"s: {s[:]}...")
            
            matches = check_model(p, q, r, s, model[:64], model[64])
            print(f"\nModel reconstruction {'successful' if matches else 'failed'}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error processing public_mod.txt: {str(e)}")

if __name__ == "__main__":
    main()
    