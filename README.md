# 🏠 Real Estate Investment Advisor

A machine learning-powered web application that helps users make informed real estate investment decisions by predicting whether a property is a good investment and estimating its future value after 5 years.

## ✨ Features

- **📊 Investment Prediction**: Classifies properties as good or bad investments using machine learning
- **💰 Price Estimation**: Predicts future property value after 5 years
- **🗺️ Multi-State Support**: Covers 19 major Indian states
- **📈 Interactive Visualizations**: Comprehensive EDA with charts and graphs
- **🎛️ User-Friendly Interface**: Simple Streamlit-based UI
- **🔬 MLflow Integration**: Model tracking and experiment management

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Tracking**: MLflow
- **Model Serialization**: Joblib

## 📁 Project Structure

```
Real Estate Investment Advisor/
├── app.py                 # Main Streamlit application
├── model/                 # Trained ML models
│   ├── classification_model.pkl
│   ├── regression_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── dataset/               # Dataset files
│   └── india_housing_prices.csv
├── env/                   # Virtual environment
├── mlruns/               # MLflow experiment tracking
├── mlflow.db             # MLflow database
├── .gitignore            # Git ignore file
├── LICENSE               # MIT License
└── README.md             # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Mini Project 2"
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv env
   # Windows
   env\Scripts\activate
   # Linux/Mac
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib mlflow
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

The application will open in your web browser at `http://localhost:8501`

## 🎯 How to Use

1. **Enter Property Details**: Use the sidebar to input property characteristics
   - BHK configuration
   - Property size (sqft)
   - Current price (in Lakhs)
   - Location details (floor, age, etc.)
   - Nearby amenities (schools, hospitals)
   - State location

2. **Get Predictions**: Click "Predict Investment Potential" to receive:
   - Investment recommendation (Good/Not Good)
   - Investment confidence percentage
   - Estimated future price after 5 years

3. **Explore Data Analysis**: View the EDA section for insights about:
   - Price distributions
   - Size vs price relationships
   - BHK distributions
   - Property type comparisons
   - Feature correlations

## 🤖 Model Details

### Features Used

- **Numerical Features**: BHK, Size, Price, Floor, Age, Schools, Hospitals
- **Calculated Features**: Price per SqFt, Year Built, Amenity Score
- **Categorical Features**: Parking, Security, State, Property Type
- **Location Features**: One-hot encoded localities

### Models

- **Classification Model**: Predicts investment potential (Good/Not Good)
- **Regression Model**: Predicts future property value after 5 years
- **Preprocessing**: StandardScaler for feature normalization

### Performance

- **Classification Accuracy**: ~85% (varies by dataset)
- **Regression R² Score**: ~0.78 (varies by dataset)
- **Features**: 584 total features including one-hot encoded categorical variables

## 📊 MLflow Integration

The project includes MLflow for experiment tracking:

```bash
# Start MLflow UI
mlflow ui

# View experiments at http://localhost:5000
```

- **Experiment Name**: Real_Estate_Investment_Advisor
- **Tracking**: Model parameters, metrics, and artifacts
- **Version Control**: Model versioning and comparison

## 🔧 Model Training

To retrain the models:

1. **Prepare Data**: Ensure dataset is in `dataset/india_housing_prices.csv`
2. **Run Training**: Execute the training notebook/script
3. **Save Models**: Models are automatically saved to `model/` directory
4. **Log to MLflow**: Training metrics are logged to MLflow

## 📈 Data Analysis Features

The app includes comprehensive EDA:

- **Price Distribution**: Histogram of property prices
- **Size vs Price**: Scatter plot showing relationship
- **BHK Distribution**: Count plot of BHK configurations
- **Property Types**: Box plots comparing different property types
- **Correlation Heatmap**: Feature correlation matrix

## 🌍 Supported States

The model supports predictions for 19 Indian states:
- Maharashtra, Delhi, Karnataka, Gujarat, Tamil Nadu
- Uttar Pradesh, West Bengal, Rajasthan, Madhya Pradesh
- Andhra Pradesh, Telangana, Kerala, Punjab, Haryana
- Bihar, Assam, Odisha, Jharkhand, Chhattisgarh, Uttarakhand

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🚨 Limitations

- **Data Dependency**: Model accuracy depends on training data quality
- **Market Factors**: Doesn't account for real-time market changes
- **Geographic Scope**: Limited to Indian states in training data
- **Economic Factors**: External economic factors not considered

## 🔮 Future Enhancements

- [ ] Real-time market data integration
- [ ] Additional cities and localities
- [ ] Economic indicators integration
- [ ] Mobile application
- [ ] API endpoints for integration
- [ ] Advanced visualization dashboard

## 📞 Contact

For questions or suggestions, please open an issue in the repository.

---

**⚠️ Disclaimer**: This tool provides estimates based on historical data and should not be considered as financial advice. Always consult with real estate professionals before making investment decisions.
