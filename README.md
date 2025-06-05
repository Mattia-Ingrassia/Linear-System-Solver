# Linear System Solver

This library contains a benchmarking tool for comparing different iterative linear algebra solvers on sparse matrix systems. 
The iterative methods analyzed are:

- Jacobi
- Gauss-Seidel
- Gradient
- Conjugate Gradient

## Features

- **Flexible Input Formats**: Support for .mtx matrices and various vector formats (.npy, .txt, .csv)
- **Comprehensive Analysis**: Performance comparison across different tolerance levels
- **Visual Output**: Automated generation of comparison plots and sparsity pattern visualizations
- **Detailed Results**: JSON output with convergence data, iterations, execution time, and relative errors

---

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/Mattia-Ingrassia/Linear-System-Solver.git
    cd Linear-System-Solver
    ```

2. Create a virtual environment:

    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:

   - On Windows:

        ```sh
        venv\Scripts\activate
        ```

   - On macOS and Linux:

        ```sh
        source venv/bin/activate
        ```

4. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```
---

## Project Structure

```
├── cli_interface.py           # Main CLI application
├── plot_generator.py          # Visualization utilities
├── solvers/
│   ├── base_solver.py         # Base solver interface
│   ├── jacobi.py              # Jacobi method implementation
│   ├── gauss_seidel.py        # Gauss-Seidel method implementation
│   ├── gradient.py            # Gradient method implementation
│   └── conjugate_gradient.py  # Conjugate Gradient implementation
└── results/                   # Generated results directory
```

## Usage

### Basic Usage

```bash
python cli_interface.py --a_path matrix.mtx
```

### Advanced Usage

```bash
python cli_interface.py \
    --a_path matrix.mtx \
    --b_path rhs_vector.txt \
    --x_path solution_vector.txt \
    --tol 1e-4 --tol 1e-6 --tol 1e-8 \
    --verbose True
```

### Command Line Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--a_path` | Path | Yes | Path to coefficient matrix in .mtx format |
| `--b_path` | Path | No | Path to right-hand side vector (.npy, .txt, .csv) |
| `--x_path` | Path | No | Path to true solution vector (.npy, .txt, .csv) |
| `--tol` | Float | No | Tolerance values (can be specified multiple times) |
| `--verbose` | Bool | No | Enable verbose output (default: True) |

### Default Behavior

- **Missing b vector**: Calculated as `b = A * x`
- **Missing x vector**: Initialized as a vector of ones
- **Missing tolerances**: Uses default values `[1e-4, 1e-6, 1e-8, 1e-10]`

## Supported File Formats

### Matrix Format
- **Input**: `.mtx` (Matrix Market format)
- **Processing**: Converted to dense format for computation

### Vector Formats
- **`.npy`**: NumPy binary format
- **`.txt`**: Plain text (space-separated values)
- **`.csv`**: Comma-separated values

## Output

### Directory Structure
```
results/
└── matrix_name/
    ├── results.json             
    ├── matrix_name_sparsity_pattern.png
    └── images/
        ├── matrix_name_iterations_comparison.png
        ├── matrix_name_time_comparison.png
        └── matrix_name_error_comparison.png
```

### Results JSON Format
```json
[
    {
        "tolerance": 10e-4,
        "solutions": [
            {
                "solution" : [],
                "relative_error": 0.01758105433208519,
                "iterations": 82,
                "time_spent": 0.027469000007840805,
                "solver_name": "Jacobi",
                "converged": true
            }
        ]
    }
]
```

### Generated plots

1. **Iterations Comparison**: Number of iterations vs tolerance for each solver
2. **Time Comparison**: Execution time vs tolerance for each solver  
3. **Error Comparison**: Relative error vs tolerance for each solver
4. **Sparsity Pattern**: Visual representation of matrix structure and density
