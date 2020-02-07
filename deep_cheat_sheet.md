# Deep Cheat Sheet

- the [recipe](http://karpathy.github.io/2019/04/25/recipe/)

## Classification

- Use crossentropy loss for probabilities

### Binary Classification
- Use a `sigmoid` activation on the last layer
- Use `binary_crossentropy` loss

### Multiclass Classification
- Use a `softmax` activation on the last layer
- Use `categorical_crossentropy` loss
