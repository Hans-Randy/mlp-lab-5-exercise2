import torch


def main() -> None:
    # Number of samples in a batch (arbitrary; demonstrates the leading dimension)
    batch_size: int = 8

    # Input matrix X has one row per sample and one column per input feature
    X: torch.Tensor = torch.randn(batch_size, 10)

    # Hidden layer parameters (chosen so that X @ Wh is valid):
    #    Wh is shaped (in_features, hidden_units) = (10, 50)
    #    bh is a length-50 bias vector (broadcasted across the batch)
    Wh: torch.Tensor = torch.randn(10, 50)
    bh: torch.Tensor = torch.randn(50)

    # Output layer parameters:
    #    Wo is shaped (hidden_units, out_features) = (50, 3)
    #    bo is a length-3 bias vector
    Wo: torch.Tensor = torch.randn(50, 3)
    bo: torch.Tensor = torch.randn(3)

    # Forward pass with ReLU activation in the hidden layer
    H: torch.Tensor = torch.relu(X @ Wh + bh)  # shape: (batch_size, 50)
    Y: torch.Tensor = H @ Wo + bo              # shape: (batch_size, 3)


    print("shape of X:", tuple(X.shape))
    print("shapes of Wh and bh:", tuple(Wh.shape), tuple(bh.shape))
    print("shapes of Wo and bo:", tuple(Wo.shape), tuple(bo.shape))
    print("shape of Y:", tuple(Y.shape))
    print("Equation: Y = ReLU(X @ Wh + bh) @ Wo + bo")


if __name__ == "__main__":
    main()


