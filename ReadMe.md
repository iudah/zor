# Zor - Tensor Library

Zor is a C-based tensor library that is part of my six-module ML stack. This project was born from my struggle with TensorFlow 2 and Keras on Android, driving me to learn the inner workings of these frameworks from scratch. Zor is designed with flexibility in mind so that it can eventually interface with other tensor libraries if needed.

**Highlights:**
- Integral module in a six-part ML stack.
- Aims to support beginner-friendly ML development on Android/embedded systems.
- Built using CMake.
- Uses pcg-c as the random number generator.
- Inspired by established libraries like llamacpp, TensorFlow for embedded systems, and TinyML.

## ML Stack Modules
1. **memalloc:** Memory allocator/manager.
2. **zot:** Dummy interface for memalloc.
3. **zobject:** Object-oriented programming emulator.
4. **zor:** Tensor library (this project).
5. **zode:** Autodiff library.
6. **zone:** Neural network library.

## Getting Started

Clone and build with CMake:
```bash
git clone https://github.com/iudah/zor.git
cd zor
cmake .
make
```

## Examples

Explore the examples folder to see Zor in action.

## Feedback

For any questions or suggestions, please open an issue.

