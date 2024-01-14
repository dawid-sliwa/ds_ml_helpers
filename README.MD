
## Functions

### 1. `walk_through_dir(dir_path: str)`
Walks through a directory and prints the number of subdirectories and images in each subdirectory.
- `dir_path`: Path of the directory to walk through.

### 2. `plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor)`
Plots the decision boundary of a model based on provided data.
- `model`: PyTorch model.
- `X`: Input tensor.
- `y`: Labels tensor.

### 3. `plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None)`
Plots training and test data, and optionally, the predictions.
- `train_data`, `train_labels`: Training data and labels.
- `test_data`, `test_labels`: Test data and labels.
- `predictions`: Model predictions, optional.

### 4. `accuracy_fn(y_true, y_pred)`
Calculates the accuracy between true labels and predictions.
- `y_true`: Truth labels tensor.
- `y_pred`: Predictions tensor.

### 5. `print_train_time(start, end, device=None)`
Prints the time taken for training.
- `start`, `end`: Start and end times.
- `device`: Computation device, optional.

### 6. `plot_loss_curves(results)`
Plots the loss and accuracy curves from training results.
- `results`: Dictionary containing training and test loss and accuracy.

### 7. `pred_and_plot_image(model, image_path, class_names=None, transform=None, device)`
Makes a prediction on an image and plots it.
- `model`: Trained PyTorch model.
- `image_path`: Path to the target image.
- `class_names`: Class names, optional.
- `transform`: Image transform, optional.
- `device`: Computation device.

### 8. `set_seeds(seed=42)`
Sets random seeds for reproducibility.
- `seed`: Seed value, default is 42.

### 9. `download_data(source, destination, remove_source=True)`
Downloads and unzips data from a source URL.
- `source`: URL to download data from.
- `destination`: Local destination path.
- `remove_source`: Flag to remove source file after extraction, default is True.

## Example Usage

```python
# Example: Using walk_through_dir function
walk_through_dir("path/to/directory")

# Example: Setting seeds
set_seeds(123)

# Example: Downloading and preparing data
data_path = download_data("http://source.to/data.zip", "data_folder")

# Example: Plotting decision boundary
plot_decision_boundary(model, X_train, y_train)

# Example: Calculating accuracy
acc = accuracy_fn(y_true, y_pred)
print(f"Accuracy: {acc}%")
```