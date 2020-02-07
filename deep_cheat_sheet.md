# Deep Cheat Sheet

- use the [recipe](http://karpathy.github.io/2019/04/25/recipe/)!
- "If you can't come up with a simple heuristic after looking at the data, perhaps AI is not the right solution for your problem"

## Data Preparation

- normalize structured data to have 0 mean & std dev
```
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std
```
- splits: train, val, test

## Model Architecture

- `relu` > `tanh` activation
- bigger network is not necessarily better
- small training set -> more overfitting (mitigate with smaller network)

## Optimization

- larger batch size is generally always better
- with very small datasets, use K-fold validation (typically K = 4 or 5)
- with small datasets, once the optimal architecture / hyperparameters have been found, train final model on combined train + val dataset

### Classification

- Use crossentropy loss for probabilities
- metrics: `accuracy`

#### Binary Classification
- Use a `sigmoid` activation on the last layer
- Use `binary_crossentropy` loss

#### Multiclass Classification
- Use a `softmax` activation on the last layer
- Use `categorical_crossentropy` loss with one hot vector labels
- Use `sparse_categorical_crossentropy` loss with integer encoding labels

### Regression
- Use NO activation on the last layer
- Use `mse` loss
- metrics: `mae` (mean absolute error)
