# CloseAuctionPredict

This repository contains all the code developed for my Master's thesis:  
**[Forecasting Closing Auction Price Movements Using Data-Driven Methods](https://dspace.uba.uva.nl/server/api/core/bitstreams/79f2dd30-d51f-489c-b2d7-710ecdbe66a0/content)**

As the title suggests, the project explores various data-driven approaches, particularly deep learning models, to predict stock price movements in the closing auction phase of European markets. 

The models developed include Linear Regression, 1D Convolutional Neural Networks, Long Short-Term Memory networks (LSTMs) and transformer models, all trained to predict return directions in the closing auction. 

The project also evaluates how predictive performance changes when using different types of information, including limit order book data, trade data and previous-day returns (referred to as exogenous variables).

## Project Structure & Data Flow
All source code is contained in the `src/` directory. It is organized into modular subpackages, each responsible for a key component of the research pipeline.
### Submodules
1. **`DataProcessing/`**  
   Handles the full data pipeline, transforming raw compressed CSV files into structured input features formatted as PyTorch tensors. It consists of three main pipeline modules:
   - **Processors**: Clean and preprocess raw data, including splitting by day and filtering out inconsistencies (e.g., abnormally high prices).
   - **FeatureEngineers**: Transform the cleaned data into feature-rich representations suitable for modeling.
   - **DatasetAssemblers**: Combine engineered features into finalized datasets ready to be fed into models.

2. **`Layers/`**  
   Contains custom-defined layers used within the different model architectures. These include, for example, positional encoding layers for Transformers and reshaping layers (e.g., unsqueeze) used in CNNs to convert convolutional outputs into vectors suitable for fully connected layers.

3. **`Metrics/`**  
   Contains custom evaluation metrics relevant to the experimental setup. Standard metrics like accuracy and binary cross-entropy are provided by existing libraries, so this submodule focuses on profit-related metrics. These are implemented within a single class, `ProfitCalculator`, which computes various trading-oriented performance measures.

4. **`Modeling/`**  
   Responsible for managing the full modeling pipeline using the constructed datasets and defined model architectures. This submodule consists of three main components:
   - **`DatasetManagers/`**: Load and prepare datasets, including splitting into training, validation, and test sets.
   - **`HyperOptimizer/`**: Performs hyperparameter optimization across different model types and configurations.
   - **`ModelRunner/`**: Coordinates model training, validation, and evaluation by combining a dataset manager and a model instance.
  
5. **`Models/`**  
   Contains all model architectures used in this project. Each model type is organized into its own subfolder, which includes a base variant along with additional variants that incorporate different input types such as limit order book data, trade data, or exogenous features.

6. **`Utils/`**  
   Contains utility functions for generating mock data. These are primarily used for testing and debugging pipeline components without relying on real datasets. Since the actual dataset used in this project is confidential and cannot be shared, these mock data generators are also used in example scripts to demonstrate functionality and structure.

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/Asewlal01/CloseAuctionPredict.git
cd CloseAuctionPredict
```
### 1.1 Setting up a virtual environment (Recommended)
This project requires Python and can be run in any standard virtual environment. While it is possible to run the project using your global Python environment, this is not recommended to avoid potential package version inconsistencies that could affect reproducibility. 

For example, using Python's built-in `venv`, we can setup and activate the environment as follows:
```bash
python -m venv venv
source venv/bin/activate 
```

### 2. Install dependencies
This project uses a `pyproject.toml` file for dependency and metadata management. You can install the project and its dependencies using standard `pip`. If you **do not plan to modify the source code**, install the package normally:
```bash
pip install .
```
If you intend to modify or extend the source code, install the package in edit mode:
```bash
pip install -e .
```

#### Note on PyTorch Installation

In this project, PyTorch is installed using the standard PyPI package (`pip install torch`). However, depending on your operating system and whether you are using a GPU, you may need to install a different version of PyTorch

If you encounter errors during installation or want to enable GPU acceleration, please refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) to select the correct command for your system.

## Example Scripts
The `example_scripts/` folder contains example scripts that demonstrate how to use the codebase with mock data. Additionally, they can also be used to verify if everything works as intended.

### 1. Creating Mock Data
Since the actual dataset used in this project is confidential and cannot be shared, randomly generated mock data is provided to allow you to run the example scripts and explore the full pipeline.

To generate the mock dataset, run:

```bash
python example_scripts/generate_data.py
```


### 2. Preprocessing
Once the data has been generated, we can perform the first step of our data pipeline: preprocessing the data. In the original dataset, this involved raw trade data and one-minute snapshots of the limit order book (LOB). The preprocessing pipeline performs the following tasks:
- Filters out extreme or erroneous prices
- Splits the dataset into individual trading days
- Aggregates and aligns trade data with LOB snapshots
- Extracts auction data from the raw trade data

First, we preprocess the limit order book as follows:
```bash
python example_scripts/preprocessing/process_limit_order_book.py
```

Next, we aggregate the trade data, and align it with the limit order book
```bash
python example_scripts/preprocessing/process_trades.py
```

Lastly, we extract auction data from the trade data
```bash
python example_scripts/preprocessing/process_auction.py
```
### 3. Feature Engineering
With the cleaned and preprocessed data available, the next step is to construct features for model training and evaluation. This step transforms our aligned trade and limit order book data into a set of time-dependent and exogenous features that can be fed into our models.

Run the feature engineering script as follows:

```bash
python example_scripts/feature_engineer.py
```

### 4. Assembling the dataset
In the final step, the engineered features are assembled into a proper dataset. This includes normalization to ensure features are on comparable scales, a critical step to obtain optimal results. 

Run the dataset assembly script as follows:
```bash
python example_scripts/assemble_dataset.py
```

### 5. Training and Evaluating the Model

With the dataset assembled, you can now train and evaluate one of the available model variants.  
In the provided example, we use an LSTM model.

The script performs the following steps:
- Loads the processed dataset
- Trains the selected model
- Evaluates performance on the test set

To run a full training and evaluation cycle using the mock dataset, execute:

```bash
python example_scripts/train_evaluate.py
```

