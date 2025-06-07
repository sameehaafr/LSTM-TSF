# LSTM Time Series Forecasting for LA Air Quality

This project implements a Long Short-Term Memory (LSTM) neural network to forecast PM10 air quality levels in Los Angeles, California. The application is built using Streamlit and provides an interactive interface for users to explore predictions and model performance.

## Features

- Interactive time series predictions
- Real-time visualization of PM10 levels
- Model performance metrics
- Air quality monitoring station locations
- Historical data analysis

## Local Development

1. Clone the repository:
```bash
git clone <your-repository-url>
cd LSTM-TSF-1
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run lstm.py
```

## Deployment

This application is deployed on Streamlit Cloud. To deploy your own version:

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file (lstm.py)
6. Click "Deploy"

## Data

The application uses PM10 air quality data from EPA monitoring stations in Los Angeles for the years 2020-2022.

## Model Architecture

- Single LSTM layer with 50 units
- ELU activation function
- Dropout rate of 0.6
- L2 regularization (0.02)
- 25 training epochs

## Author

Sameeha Afrulbasha
- [Website](https://sameehaafr.github.io/sameehaafr/)
- [GitHub](https://github.com/sameehaafr)
- [LinkedIn](https://www.linkedin.com/in/sameeha-afrulbasha/)
- [Medium](https://sameehaafr.medium.com/)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 