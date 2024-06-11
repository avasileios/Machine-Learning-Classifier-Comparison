# Machine Learning Classifier Comparison

This project compares multiple machine learning classifiers on a dataset to identify the best performing model based on accuracy, F1 score, and AUC score. It uses resampling techniques to handle class imbalance and standard scaling for feature normalization.

## Project Structure

- `process_files.py`: The main script that processes the CSV files.
- `requirements.txt`: List of dependencies.
## Setup

### Prerequisites

- Python 3.x

### Installation

1. **Install the required packages:**

   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your dataset:**

   Place your CSV file (e.g., `log2.csv`) in the project directory.

2. **Run the script:**

   ```
   python process_files.py
   ```

3. **Output:**

   - The processed results and analysis will be printed in the console.
   - Plots and logs will be generated in the specified directories.

## Dependencies

- numpy
- pandas
- scikit-learn
- imbalanced-learn
- seaborn
- matplotlib

## Directory Structure

The expected directory structure is:

```
project-directory/
├── README.md
├── requirements.txt
├── process_files.py
├── log2.csv
```

## Contributing

Feel free to fork this repository and make improvements. Pull requests are welcome.

## License

This project is licensed under the MIT License.