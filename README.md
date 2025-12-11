# Mall Movement Tracking - ML Project

A comprehensive machine learning project for tracking and analyzing customer movement patterns in shopping malls. This project includes feature engineering, classification, clustering, forecasting models, and interactive dashboards.

## 🎯 Project Overview

This project analyzes customer movement data to:
- **Predict next zone visits** using classification models
- **Cluster customer behavior patterns** using unsupervised learning
- **Forecast traffic patterns** using time series models
- **Visualize insights** through interactive Streamlit dashboards
- **Serve predictions** via FastAPI endpoints

## 📁 Project Structure

```
mall-movement-tracking/
├── api/                    # FastAPI application
│   ├── app.py              # Main API application
│   ├── routers/            # API route handlers
│   ├── schemas/            # Pydantic models
│   └── services/           # Business logic
├── config/                 # Configuration files
│   ├── project_config.yaml
│   └── secrets_template.yaml
├── data/                   # Data storage
│   ├── processed/         # Cleaned and processed data
│   └── sample/            # Sample data files
├── docs/                   # Documentation
│   ├── api_docs.md
│   ├── data_dictionary.md
│   └── model_cards/        # Model documentation
├── features/               # Feature engineering
│   ├── feature_engineering.py
│   ├── feature_config.yaml
│   └── run_feature_engineering.py
├── models/                 # Trained models
│   ├── classification/
│   ├── clustering/
│   └── forecasting/
├── notebooks/              # Jupyter notebooks
│   ├── 01_EDA.ipynb       # Exploratory Data Analysis
│   ├── 02_Feature_Analysis.ipynb
│   ├── 03_Modeling_Experiments.ipynb
│   └── 04_Model_Comparison.ipynb
├── results/                # Model results and metrics
├── streamlit_app/         # Streamlit dashboard
│   ├── app.py
│   └── pages/             # Dashboard pages
├── training/              # Training scripts
│   ├── train_classification.py
│   ├── train_clustering.py
│   └── train_forecasting.py
└── tests/                  # Unit tests
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/mall-movement-tracking.git
cd mall-movement-tracking
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare data**
   - Place your processed data in `data/processed/merged data set.csv`
   - Or use the existing processed data

5. **Run feature engineering**
```bash
python features/run_feature_engineering.py
```

6. **Train models**
```bash
# Train classification models
python training/train_classification.py

# Train clustering models
python training/train_clustering.py

# Train forecasting models
python training/train_forecasting.py
```

7. **Run Streamlit dashboard**
```bash
streamlit run streamlit_app/app.py
```

8. **Run API server**
```bash
cd api
uvicorn app:app --reload
```

## 📊 Features

### Feature Engineering
- ✅ Missing value handling
- ✅ Categorical encoding (label & one-hot)
- ✅ Datetime feature extraction
- ✅ Outlier detection & handling
- ✅ Binning/grouping
- ✅ Domain-specific features
- ✅ Column combining

### Models
- **Classification**: Random Forest, Decision Tree, XGBoost
- **Clustering**: K-Means, DBSCAN
- **Forecasting**: ARIMA, Prophet

### Dashboards
- Data Explorer
- Heatmaps
- Classification Results
- Clustering Insights
- Forecasting Traffic
- Next Zone Prediction
- Model Explainability

## 📖 Usage

### Running Notebooks

1. **EDA Analysis**
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

2. **Feature Analysis**
```bash
jupyter notebook notebooks/02_Feature_Analysis.ipynb
```

### API Endpoints

Once the API is running, access:
- `http://localhost:8000/` - API root
- `http://localhost:8000/docs` - Interactive API documentation
- `http://localhost:8000/api/data/info` - Dataset information
- `http://localhost:8000/api/results/classification` - Classification results
- `http://localhost:8000/api/results/clustering` - Clustering results
- `http://localhost:8000/api/results/forecasting` - Forecasting results

## 🛠️ Configuration

Edit `features/feature_config.yaml` to customize feature engineering:
- Missing value handling strategy
- Encoding methods
- Outlier detection methods
- Binning parameters
- Domain-specific features

## 📝 Workflow

1. **Exploratory Data Analysis** → `notebooks/01_EDA.ipynb`
2. **Feature Engineering** → `notebooks/02_Feature_Analysis.ipynb` or `features/run_feature_engineering.py`
3. **Model Training** → `training/train_*.py` scripts
4. **Visualization** → Streamlit dashboard
5. **API** → FastAPI endpoints

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

## 📚 Documentation

- [API Documentation](docs/api_docs.md)
- [Data Dictionary](docs/data_dictionary.md)
- [Model Cards](docs/model_cards/)
- [Feature Engineering Guide](features/README.md)
- [Workflow Guide](WORKFLOW.md)
- [GitHub Setup Guide](GITHUB_SETUP.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Your Name - [GitHub Profile](https://github.com/YOUR_USERNAME)

## 🙏 Acknowledgments

- Libraries: pandas, scikit-learn, xgboost, streamlit, fastapi
- Data sources: [Your data source]

## 📧 Contact

For questions or suggestions, please open an issue or contact [your-email@example.com]

---

**Note**: This project uses processed/cleaned data. Large data files and model files (.pkl) are excluded from the repository. Make sure to have your data in `data/processed/` before running the project.

