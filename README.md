# Market Basket Analysis Application

This repository contains a Streamlit web application for performing market basket analysis using the Apriori algorithm. The application can analyze transaction data to discover associations between products frequently purchased together.

## Features

- Load transaction data from CSV files
- Apply the Apriori algorithm to discover frequent itemsets
- Generate association rules with support, confidence, and lift metrics
- Visualize results with multiple chart types:
  - Frequent items bar chart
  - Association network graph
  - Rules heatmap
  - Rules scatter plot

## Applications

This repository contains three different applications to handle various data formats and requirements:

### 1. Standard Market Basket Analysis (app.py)

The main application for performing market basket analysis with a standard data format.

```bash
streamlit run app.py
```

### 2. Multi-Format Market Basket Analysis (app_multi_format.py)

An enhanced version that automatically detects and handles multiple data formats.

```bash
streamlit run app_multi_format.py
```

Features:

- Automatic encoding and delimiter detection
- Support for various data formats
- Data format transformation options
- Detailed parameter explanations

### 3. Data Format Converter (app_data_converter.py)

A dedicated tool for converting various CSV formats into the standard format required for market basket analysis.

```bash
streamlit run app_data_converter.py
```

Features:

- Convert from various formats to the standard format
- Support for individual product rows, combined product cells, and one-hot encoded formats
- Preview data before and after conversion
- Download converted data as CSV
- Sample code for using the converted data

## Data Format Requirements

The applications support different data formats:

1. **Standard Format**: CSV with TransactionID, CustomerID, and Products columns, where Products contains comma-separated lists of items.

2. **Individual Products Format**: Each row represents a single product in a transaction.

3. **Combined Products Format**: Each row represents a transaction with products already combined in one cell.

4. **One-hot Encoded Format**: Each product is a separate column with boolean/binary values.

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the desired application:

```bash
streamlit run app.py
# or
streamlit run app_multi_format.py
# or
streamlit run app_data_converter.py
```

## Requirements

- Python 3.8+
- Streamlit 1.27.0
- Pandas 2.0.3
- NumPy 1.24.3
- Matplotlib 3.7.2
- Seaborn 0.12.2
- MLxtend 0.22.0
- NetworkX 3.1
- Plotly 5.16.1
- Chardet 5.2.0
