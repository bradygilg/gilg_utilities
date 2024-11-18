# gilg_utilities
A set of utility functions for machine learning.

## Installation

To install with pip,

`pip install gilg_utils @ git+https://github.com/bradygilg/gilg_utilities.git`

Optional dependencies (including pytorch) are included in pyproject.toml. To install all of the optional dependencies, run

`pip install gilg_utils[all] @ git+https://github.com/bradygilg/gilg_utilities.git`

## Usage

The models in gilg_utils.models all have four primary methods, .train, .predict, .save, and .load. A training example is
```
from gilg_utils.models import PytorchNeuralNetworkRegressor`
model = PytorchNeuralNetworkRegressor()
model.train(train_df,
            test_df=test_df,
            dimension=dimension,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            loss_function_name=loss_function_name,
            optimizer_name=optimizer_name,
            max_epochs=max_epochs,
            seed=seed,
            callback_period=callback_period)
out_path = /path/to/output/folder
model.save(out_path)
```