"""
Comprehensive Feature Engineering Module
Handles all feature engineering tasks before ML training:
- Creating new features
- Encoding categorical variables
- Handling missing values
- Binning/grouping values
- Date/time extraction
- Domain-specific transformation
- Outlier detection
- Combining or splitting columns
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import yaml

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from streamlit_app.utils.data_loader import load_processed_data

class FeatureEngineer:
    """Comprehensive feature engineering class"""
    
    def __init__(self, config_path=None):
        """
        Initialize feature engineer
        
        Args:
            config_path: Path to feature config YAML file
        """
        self.config_path = config_path or Path(__file__).parent / "feature_config.yaml"
        self.config = self._load_config()
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.imputers = {}
        self.scalers = {}
        self.bin_edges = {}
        
    def _load_config(self):
        """Load feature engineering configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def detect_column_types(self, df):
        """
        Auto-detect column types in the dataset
        
        Returns:
            dict with column categories
        """
        columns = {
            'datetime': [],
            'categorical': [],
            'numeric': [],
            'id': [],
            'target': []
        }
        
        # Detect datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        for col in df.columns:
            if col in datetime_cols:
                columns['datetime'].append(col)
            elif 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower():
                columns['datetime'].append(col)
            elif df[col].dtype == 'object':
                # Check if it's actually datetime
                try:
                    pd.to_datetime(df[col].head(100), errors='raise')
                    columns['datetime'].append(col)
                except:
                    columns['categorical'].append(col)
            elif df[col].dtype in [np.int64, np.float64]:
                columns['numeric'].append(col)
            elif 'id' in col.lower() or 'user' in col.lower() or 'customer' in col.lower():
                columns['id'].append(col)
            elif 'zone' in col.lower() or 'location' in col.lower():
                columns['categorical'].append(col)
        
        return columns
    
    def handle_missing_values(self, df, strategy='auto'):
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: 'auto', 'mean', 'median', 'mode', 'drop', 'forward_fill'
        
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if strategy == 'auto':
                    if df[col].dtype in [np.int64, np.float64]:
                        # Numeric: use median
                        imputer = SimpleImputer(strategy='median')
                        df[[col]] = imputer.fit_transform(df[[col]])
                        self.imputers[col] = imputer
                    else:
                        # Categorical: use mode
                        imputer = SimpleImputer(strategy='most_frequent')
                        df[[col]] = imputer.fit_transform(df[[col]])
                        self.imputers[col] = imputer
                elif strategy == 'mean' and df[col].dtype in [np.int64, np.float64]:
                    imputer = SimpleImputer(strategy='mean')
                    df[[col]] = imputer.fit_transform(df[[col]])
                    self.imputers[col] = imputer
                elif strategy == 'median' and df[col].dtype in [np.int64, np.float64]:
                    imputer = SimpleImputer(strategy='median')
                    df[[col]] = imputer.fit_transform(df[[col]])
                    self.imputers[col] = imputer
                elif strategy == 'mode':
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[[col]] = imputer.fit_transform(df[[col]])
                    self.imputers[col] = imputer
                elif strategy == 'forward_fill':
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                elif strategy == 'drop':
                    df = df.dropna(subset=[col])
        
        return df
    
    def extract_datetime_features(self, df, datetime_column=None):
        """
        Extract comprehensive date/time features
        
        Args:
            df: DataFrame with datetime column
            datetime_column: Name of datetime column (auto-detected if None)
        
        Returns:
            DataFrame with temporal features added
        """
        df = df.copy()
        
        # Auto-detect datetime column
        if datetime_column is None:
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                datetime_column = datetime_cols[0]
            else:
                # Try to convert common date column names
                for col in ['date', 'timestamp', 'time', 'datetime', 'Date', 'Time', 'Timestamp']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        datetime_column = col
                        break
        
        if datetime_column and datetime_column in df.columns:
            if df[datetime_column].dtype != 'datetime64[ns]':
                df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
            
            # Basic temporal features
            df['hour'] = df[datetime_column].dt.hour
            df['day_of_week'] = df[datetime_column].dt.dayofweek
            df['day_of_month'] = df[datetime_column].dt.day
            df['day_of_year'] = df[datetime_column].dt.dayofyear
            df['week'] = df[datetime_column].dt.isocalendar().week
            df['month'] = df[datetime_column].dt.month
            df['quarter'] = df[datetime_column].dt.quarter
            df['year'] = df[datetime_column].dt.year
            
            # Boolean temporal features
            df['is_weekend'] = df[datetime_column].dt.dayofweek.isin([5, 6]).astype(int)
            df['is_weekday'] = (~df[datetime_column].dt.dayofweek.isin([5, 6])).astype(int)
            df['is_morning'] = df['hour'].between(6, 11).astype(int)
            df['is_afternoon'] = df['hour'].between(12, 17).astype(int)
            df['is_evening'] = df['hour'].between(18, 23).astype(int)
            df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
            
            # Cyclical encoding for periodic features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def encode_categorical_variables(self, df, columns=None, method='auto'):
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            columns: List of columns to encode (None = auto-detect)
            method: 'auto', 'label', 'onehot', 'target'
        
        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()
        
        if columns is None:
            # Auto-detect categorical columns
            columns = df.select_dtypes(include=['object']).columns.tolist()
            # Also include low cardinality numeric columns that might be categorical
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].nunique() < 20 and df[col].nunique() < len(df) * 0.1:
                    columns.append(col)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'auto':
                # Use label encoding for high cardinality, onehot for low
                if df[col].nunique() > 10:
                    method_used = 'label'
                else:
                    method_used = 'onehot'
            else:
                method_used = method
            
            if method_used == 'label':
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                # Optionally drop original
                # df = df.drop(columns=[col])
            elif method_used == 'onehot':
                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                # Optionally drop original
                # df = df.drop(columns=[col])
        
        return df
    
    def detect_and_handle_outliers(self, df, columns=None, method='iqr'):
        """
        Detect and handle outliers
        
        Args:
            df: Input DataFrame
            columns: List of numeric columns to check (None = all numeric)
            method: 'iqr', 'zscore', 'isolation'
        
        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                # Cap values beyond 3 standard deviations
                df.loc[z_scores > 3, col] = df[col].mean() + 3 * np.sign(df.loc[z_scores > 3, col] - df[col].mean()) * df[col].std()
        
        return df
    
    def create_bins(self, df, columns=None, n_bins=5, method='quantile'):
        """
        Create bins/group values
        
        Args:
            df: Input DataFrame
            columns: List of columns to bin (None = auto-select numeric)
            n_bins: Number of bins
            method: 'quantile', 'uniform', 'kmeans'
        
        Returns:
            DataFrame with binned features
        """
        df = df.copy()
        
        if columns is None:
            # Auto-select numeric columns with sufficient unique values
            columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                      if df[col].nunique() > n_bins]
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'quantile':
                df[col + '_binned'] = pd.qcut(df[col], q=n_bins, duplicates='drop', labels=False)
                self.bin_edges[col] = pd.qcut(df[col], q=n_bins, duplicates='drop').categories
            elif method == 'uniform':
                df[col + '_binned'] = pd.cut(df[col], bins=n_bins, duplicates='drop', labels=False)
                self.bin_edges[col] = pd.cut(df[col], bins=n_bins, duplicates='drop').categories
        
        return df
    
    def create_domain_features(self, df, zone_column=None, user_column=None):
        """
        Create domain-specific features for mall movement tracking
        
        Args:
            df: Input DataFrame
            zone_column: Name of zone/location column
            user_column: Name of user/customer column
        
        Returns:
            DataFrame with domain features added
        """
        df = df.copy()
        
        # Auto-detect columns
        if zone_column is None:
            zone_cols = [col for col in df.columns if 'zone' in col.lower() or 'location' in col.lower()]
            zone_column = zone_cols[0] if zone_cols else None
        
        if user_column is None:
            user_cols = [col for col in df.columns if 'user' in col.lower() or 'customer' in col.lower() or 'id' in col.lower()]
            user_column = user_cols[0] if user_cols else None
        
        if zone_column and user_column:
            # Zone visit counts per user
            zone_counts = df.groupby([user_column, zone_column]).size().reset_index(name='visit_count')
            df = df.merge(zone_counts, on=[user_column, zone_column], how='left')
            
            # Total zones visited per user
            user_zone_counts = df.groupby(user_column)[zone_column].nunique().reset_index(name='total_zones_visited')
            df = df.merge(user_zone_counts, on=user_column, how='left')
            
            # Average visits per zone per user
            df['avg_visits_per_zone'] = df['visit_count'] / df['total_zones_visited'].replace(0, 1)
            
            # Zone popularity (total visits across all users)
            zone_popularity = df.groupby(zone_column).size().reset_index(name='zone_popularity')
            df = df.merge(zone_popularity, on=zone_column, how='left')
            
            # User activity level (total visits)
            user_activity = df.groupby(user_column).size().reset_index(name='user_activity_level')
            df = df.merge(user_activity, on=user_column, how='left')
        
        return df
    
    def combine_columns(self, df, combinations=None):
        """
        Combine or split columns
        
        Args:
            df: Input DataFrame
            combinations: List of tuples (col1, col2, operation, new_name)
        
        Returns:
            DataFrame with combined columns
        """
        df = df.copy()
        
        if combinations is None:
            # Auto-create some common combinations for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                # Sum of first two numeric columns
                if len(numeric_cols) >= 2:
                    df[f'{numeric_cols[0]}_plus_{numeric_cols[1]}'] = df[numeric_cols[0]] + df[numeric_cols[1]]
                    df[f'{numeric_cols[0]}_mult_{numeric_cols[1]}'] = df[numeric_cols[0]] * df[numeric_cols[1]]
        else:
            for col1, col2, operation, new_name in combinations:
                if col1 in df.columns and col2 in df.columns:
                    if operation == 'add':
                        df[new_name] = df[col1] + df[col2]
                    elif operation == 'multiply':
                        df[new_name] = df[col1] * df[col2]
                    elif operation == 'divide':
                        df[new_name] = df[col1] / (df[col2].replace(0, np.nan))
                    elif operation == 'subtract':
                        df[new_name] = df[col1] - df[col2]
                    elif operation == 'concat':
                        df[new_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
        
        return df
    
    def engineer_features(self, df, config=None):
        """
        Main feature engineering pipeline
        
        Args:
            df: Raw processed DataFrame
            config: Optional config dict to override defaults
        
        Returns:
            DataFrame with all engineered features
        """
        print("Starting feature engineering pipeline...")
        df = df.copy()
        
        # Use provided config or class config
        cfg = config or self.config
        
        # 1. Handle missing values
        print("  - Handling missing values...")
        missing_strategy = cfg.get('missing_values', {}).get('strategy', 'auto')
        df = self.handle_missing_values(df, strategy=missing_strategy)
        
        # 2. Extract datetime features
        print("  - Extracting datetime features...")
        datetime_col = cfg.get('datetime', {}).get('column')
        df = self.extract_datetime_features(df, datetime_column=datetime_col)
        
        # 3. Encode categorical variables
        print("  - Encoding categorical variables...")
        encode_method = cfg.get('encoding', {}).get('method', 'auto')
        encode_cols = cfg.get('encoding', {}).get('columns')
        df = self.encode_categorical_variables(df, columns=encode_cols, method=encode_method)
        
        # 4. Detect and handle outliers
        print("  - Detecting and handling outliers...")
        outlier_method = cfg.get('outliers', {}).get('method', 'iqr')
        outlier_cols = cfg.get('outliers', {}).get('columns')
        df = self.detect_and_handle_outliers(df, columns=outlier_cols, method=outlier_method)
        
        # 5. Create bins
        print("  - Creating bins...")
        if cfg.get('binning', {}).get('enabled', False):
            n_bins = cfg.get('binning', {}).get('n_bins', 5)
            bin_method = cfg.get('binning', {}).get('method', 'quantile')
            bin_cols = cfg.get('binning', {}).get('columns')
            df = self.create_bins(df, columns=bin_cols, n_bins=n_bins, method=bin_method)
        
        # 6. Create domain-specific features
        print("  - Creating domain-specific features...")
        zone_col = cfg.get('domain', {}).get('zone_column')
        user_col = cfg.get('domain', {}).get('user_column')
        df = self.create_domain_features(df, zone_column=zone_col, user_column=user_col)
        
        # 7. Combine columns
        print("  - Combining columns...")
        if cfg.get('combining', {}).get('enabled', True):
            combinations = cfg.get('combining', {}).get('combinations')
            df = self.combine_columns(df, combinations=combinations)
        
        print(f"Feature engineering complete! Shape: {df.shape}")
        return df


# Convenience function for backward compatibility
def engineer_features(df, config_path=None):
    """
    Main feature engineering function (backward compatible)
    
    Args:
        df: Raw processed DataFrame
        config_path: Path to feature config YAML file
    
    Returns:
        DataFrame with engineered features
    """
    fe = FeatureEngineer(config_path=config_path)
    return fe.engineer_features(df)
