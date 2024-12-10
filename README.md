# Linear Recurrence Library (linrec)

This is a simple demonstration of the applications of linear algebra in recommender systems. is a Python project showcasing how linear algebra concepts can be applied to recommender systems. This project features two MVP modules:


- **`linrec_plus`**: A C++20 pybind11 module with some implementations.
- **`linrec_py`**: A Python module providing some implementations and wrapping the c++ implementations.

The library demonstrates how linear algebra techniques, including a basic implementation of `SVD++`, can be effectively used in machine learning and recommender systems.

---

## Usage

### Prerequisites

- **Python**: >= 3.7
- **C++ Compiler**: Supporting C++20 (e.g., GCC 10+, Clang 10+)
- **CMake**: >= 3.16
- **Ninja**: For efficient builds
- **Pybind11**: For C++ and Python bindings
- **Eigen**: Linear algebra library for C++

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/bm-coelho/alglin-trab.git
   cd alglin-trab
   ```

2. Build the C++ module:
   ```bash
   just build
   ```

---

## Usage

### Jupyter Demonstration

Run the `demo.ipynb` notebook to explore practical applications:

```bash
jupyter notebook demo.ipynb
```

---

## Project Structure

```text
alglin-trab/
├── alglin_py/          # Python utilities
├── alglin_plus/        # C++ module (bindings and implementations)
├── demo.ipynb          # Jupyter notebook demonstration
├── Justfile            # Build automation script
└── README.md           # Project documentation
```

