# Recommendation Impact Analysis Tool

This Streamlit application analyzes how different recommendations affect runtime performance in optimization scenarios. The tool fetches optimization data, processes it, and visualizes the impact of various recommendations on runtime performance.

## Features

* **Data Processing**: Fetches optimization data using the Falcon API client
* **Statistical Analysis**: Fits Ordinary Least Squares (OLS) and Bayesian Ridge Regression linear regression models using statsmodels and scikit-learn to evaluate the impact of different recommendations
* **Interactive Visualization**: Displays the impact of recommendations on runtime with interactive Plotly charts

![Dashboard](dashboard.png)

## Requirements

- Python 3.11 or higher
- Dependencies listed in requirements.txt

## Getting Started



### Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Add your Thanos username and password to the .env file

   A default .env file is provided targeting the production environment, but this is liable to become outdated.

### Running the Application

```
streamlit run streamlit_app.py
```

## Usage

1. Enter your Optimization ID in the sidebar
2. Verify the path to your environment file
3. Use the sidebar to configure the confidence level and statistical method
4. Click "Process Optimization" to analyze the data
5. Explore the impact of different recommendations on runtime performance

# Design

This application uses the following pipeline to retrieve, transform and analyse optimisation data:

## 1. Data Retrieval
- **Source:** Falcon API via `FalconClientWrapper`
- **Process:** Retrieves optimization solutions data using credentials from an environment file
- **Output:** `SolutionsResponse` containing validated data.

## 2. Data Transformation
- **Function:** `transform()`
- **Process:** Converts the raw `SolutionsResponse` into a tabular format (`SolutionsTabular`) ready for analysis

## 3. Statistical Analysis
- **Method Selection:** Based on a specified method (default: Ordinary Least Squares)
- **Factory Pattern:** `get_model_fitter()` creates the appropriate model fitter
- **Process:** Fits the selected statistical model to the tabular data and generates a `ModelAdapter`

## 4. Analysis
- **Process:** Uses the `ModelAdapter` to generate an `AnalysisResult` at the specified confidence level

## Data Flow

![Data Flow](data_flow.png)