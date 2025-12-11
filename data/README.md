# Data Directory

This directory contains processed and cleaned data for the mall movement tracking project.

## Structure

- `processed/` - Contains the cleaned and merged dataset
  - `merged data set.csv` - Main processed dataset (CSV format)
  - `merged data set.xlsx` - Main processed dataset (Excel format)
  
- `sample/` - Reserved for sample data files

## Usage

The processed data is automatically loaded by the data loader utility:
```python
from streamlit_app.utils.data_loader import load_processed_data

df = load_processed_data()  # Loads merged data set.csv
```

## Note

Raw data folder has been removed as ETL is complete. All models and applications use the processed/cleaned data from the `processed/` folder.

