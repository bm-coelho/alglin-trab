# Linear Recurrence Library (linrec)

This is a simple demonstration of the applications of linear algebra in recommender systems. is a Python project showcasing how linear algebra concepts can be applied to recommender systems. This project features two MVP modules:


- **`linrec_plus`**: A C++20 pybind11 module with some implementations.
- **`linrec_py`**: A Python module providing some implementations and wrapping the c++ implementations.

The library demonstrates how linear algebra techniques, including a basic implementation of `SVD++`, can be effectively used in machine learning and recommender systems.

---

## Features

- **Linear Recurrence Computations**: Fast and efficient computations using the C++20 `linrec_plus` module.
- **Utilities**: Python utilities for preprocessing and normalization.
- **Integration**: A high-level Python interface for seamless usage.
- **Jupyter Demonstration**: A `demo.ipynb` notebook to illustrate practical applications.

---

## Installation

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
   git clone https://github.com/yourusername/linrec.git
   cd linrec
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Build the C++ module:
   ```bash
   just build
   ```

4. Install the Python library:
   ```bash
   just install
   ```

---

## Usage

### Example: Linear Recurrence

Using the core library to compute a linear recurrence:

```python
from linrec import linear_recurrence

coeffs = [1, -0.5]
initial = [1, 1]
terms = 10

result = linear_recurrence(coeffs, initial, terms)
print("Generated Sequence:", result)
```

### Jupyter Demonstration

Run the `demo.ipynb` notebook to explore practical applications:

```bash
jupyter notebook demo.ipynb
```

---

## Project Structure

```text
linrec/
├── linrec_core/        # Core Python module
├── linrec_py/          # Python utilities
├── linrec_plus/        # C++ module (bindings and implementations)
├── demo.ipynb          # Jupyter notebook demonstration
├── setup.py            # Python package setup
├── justfile            # Build automation script
└── README.md           # Project documentation
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve this library.

---

## Acknowledgments

This library uses:
- [Eigen](https://eigen.tuxfamily.org): For linear algebra in C++
- [Pybind11](https://pybind11.readthedocs.io): For C++ and Python integration

---

Enjoy exploring the intersection of linear algebra and recommender systems!

