# Artificial Neural Network Library

This is a _personal_ project - my attempt to create a fully functioning, optimized, and simple AI library in pure C++ (no dependencies).

## Installation

Install with `cmake`:

```bash
  cmake -S . -B build \
        -D CMAKE_BUILD_TYPE=Release \   # or Debug
        -D PARALLEL_COST_MINIMUM=10000

  cmake --build build
```

And executable should exist in `./build/src/NeuralNetwork_exec`

### CMake Variables

- `CMAKE_BUILD_TYPE` - pretty straightforward. `Release` or `Debug`.
- `PARALLEL_COST_MINIMUM` - minimum number of operations per iteration to justify parallelizing work. It should be in "integer addition units".

### MacOS Device Warning

On MacOS devices, when using the library or running the executable, the following OS warning message may appear:

```
 malloc: nano zone abandoned due to inability to reserve vm space.
```

This warning appears to be harmless. To suppress it, set the following environment variable:

```shell
export MallocNanoZone='0'
```

## Features

1. **Core Layer Types**

- **Layers**: Dense, Dropout
- **Activations**: Step, ReLU, Leaky ReLU, Sigmoid, Softmax

2. **Loss Functions**

- **Classification**: Categorical Cross-Entropy, Categorical Cross-Entropy + Softmax, Binary Cross-Entropy
- **Regression**: Mean Absolute Error, Mean Squared Error

3. **Optimizers**

- SGD, AdaGrad, RMSProp, Adam

4. **Data loaders**

- CSV Loader
- MNIST Loader

5. **Feed Forward Model class**

- Configurable via model descriptors (see [layerDescriptors.h](include/ann/layerDescriptors.h))
- Supports training, evaluation, and prediction
- Save/Load trainable parameters
- Loading full model configuration from a custom configuration (for format guidelines, see [example.model](example.model))

## Demo

For usage demo, see [main.cpp](src/main.cpp)

## License

[GNU GPL v3](LICENSE)

## Authors

- [@bar2011](https://www.github.com/bar2011) (me)
