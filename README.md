# Neural Networks Project

So this is my attempt at creating a neural network library from scratch (no dependencies)...
This is purely a personal project, and not anything which should be taken too seriously.

# Building

```shell
cmake -S . -B build
cmake --build build
```

The executable for the main program now exists as `./build/src/NeuralNetwork_exec`, which you can simply run and everything should work.

## MacOS device warning
On MacOS devices, when using the library or running the executable, the following OS warning message may appear:
```
 malloc: nano zone abandoned due to inability to reserve vm space.
```

To remove the warning (from my research it seems meaningless), simply set the following environment variable:
```shell
export MallocNanoZone='0'
```
