import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset  # Assuming you have this dataset class
from Model import Model
from smallModel import smallModel
from util import compute_accuracy

# Define the main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model for audio denoising')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test data')
    parser.add_argument('--train_labels_path', type=str, required=True, help='Path to the training labels')
    parser.add_argument('--test_labels_path', type=str, required=True, help='Path to the test labels')
    parser.add_argument('--is_small_model', type=bool, default=True, help='Flag for small model or large model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for optimizer')
    parser.add_argument('--save_path', type=str, default='model.pth', help='Path to save the model')
    parser.add_argument('--plot_path', type=str, default='training_vs_test_loss.png', help='Path to save the plot')

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloaders
    train_data = CustomDataset(args.train_path, args.train_labels_path)
    test_data = CustomDataset(args.test_path, args.test_labels_path)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True)

    # Initialize the model (small model or not)
    if args.is_small_model:
        model = smallModel().to(device)
        print("Training small model...")
    else:
        model = Model().to(device)
        print("Training large model...")

    # Set the model to train mode
    model.train()

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.7, 0.9))

    # Loss functions (one for small models, two for larger models)
    loss_fn = nn.L1Loss()
    if not args.is_small_model:
        loss1_fn = nn.L1Loss()
        loss2_fn = nn.L1Loss()

    # For saving the best model
    train_loss = []
    test_loss = []

    # Epochs
    EPOCHS = args.epochs

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        for train_sample, train_label in train_loader:
            train_sample, train_label = train_sample.to(device), train_label.to(device)

            optimizer.zero_grad()

            # Forward pass
            if args.is_small_model:
                out = model(train_sample)
                loss = loss_fn(out, train_label)  # Only one output
            else:
                out1, out2 = model(train_sample)
                loss1 = loss1_fn(out1, train_label)
                loss2 = loss2_fn(out2, train_label)
                loss = loss1 + loss2  # Combined loss for two outputs

            # Backpropagate gradients
            loss.backward()

            # Do gradient step
            optimizer.step()

        print(f"EPOCH {epoch}, loss = {loss.item():.4f}")
        train_loss.append(loss.item())

        # Test the model after each epoch
        loss_test = compute_accuracy(model, test_loader, two_outputs=not args.is_small_model)
        test_loss.append(loss_test)

        # Save the model if it has the best test loss
        if loss_test == min(test_loss):
            torch.save(model.cpu().state_dict(), args.save_path)

        # Visualize the training and test loss after every epoch
        # plot_loss(train_loss, test_loss, args.plot_path)

    # End of training
    print("Training complete.")

#
# def plot_loss(training_loss, test_loss, plot_path):
#     """ Visualize and save the training and test loss plot """
#     fig, ax = plt.subplots(1)
#
#     # Convert losses to NumPy arrays for plotting
#     training_vis = [torch.Tensor([t]).cpu().detach().numpy() for t in training_loss]
#     test_vis = [torch.Tensor([t]).cpu().detach().numpy() for t in test_loss]
#
#     # Get minimum test loss
#     test_min = min(test_vis)
#
#     # Plotting
#     ax.axhline(y=test_min, linestyle='--', color='r')  # Line for minimum test loss
#     ax.set_yscale('log')  # Set log scale for y-axis
#     ax.plot(training_vis, label='Train Error')
#     ax.plot(test_vis, label='Test Error')
#
#     # Labeling
#     ax.set_title("Training vs Test Loss")
#     ax.set_xlabel("Epochs")
#     ax.set_ylabel("Loss")
#     ax.legend()
#
#     # Save plot as an image file
#     plt.savefig(plot_path)
#     plt.close()


if __name__ == "__main__":
    main()
