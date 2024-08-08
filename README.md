# Flower Classification with ResNet-50

This project involves building and training a ResNet-50 model to classify flower images into various categories. The project includes two main scripts:

- `main.py`: The initial, more comprehensive implementation for training the ResNet-50 model.
- `light.py`: An optimized and efficient version of the training script that uses a subset of data and includes early stopping.

## Project Structure

```
flower_classification/
├── data/                  # Directory to store dataset
│   └── flowers_data/      # Image data directory (train and val folders)
├── main.py                # Initial comprehensive training script
├── light.py               # Optimized training script
├── requirements.txt       # Required Python packages
└── README.md              # This file
```

## Requirements

To run this project, you need to have the following Python packages installed:

- torch
- torchvision
- numpy
- scipy
- pillow

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```
torch
torchvision
numpy
scipy
pillow
```

## Data Preparation

1. **Download the dataset**: Ensure you have the flower image dataset structured in the following format:

    ```
    flowers_data/
    ├── train/
    ├── val/
    ```

2. **Place your dataset**: Put your dataset in the `data/flowers_data/` directory.

3. **MATLAB Files**: Ensure that `imagelabels.mat` and `setid.mat` are in the `data/` directory if needed for data processing (as described in `main.py`).

## Usage

### Running `main.py`

This script provides a comprehensive approach to training the ResNet-50 model. It includes all the necessary steps to prepare data, configure the model, and train it. 

To run `main.py`, use:

```bash
python main.py
```

### Running `light.py`

This script provides a more efficient version of the training process by using a subset of data and incorporating early stopping. It is optimized for faster results and reduced computational load.

To run `light.py`, use:

```bash
python light.py
```

## Script Details

### `main.py`

- **Purpose**: Comprehensive training of ResNet-50 with full dataset.
- **Features**:
  - Full dataset usage
  - Standard training process without optimization techniques

### `light.py`

- **Purpose**: Optimized training of ResNet-50 using a data subset and early stopping.
- **Features**:
  - Smaller subset of data for faster training
  - Early stopping to prevent overfitting and save computational resources

## Notes

- Ensure that the data directory structure matches the expected format for the scripts to work correctly.
- Adjust the `subset_ratio` in `light.py` if you need a different subset size.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

