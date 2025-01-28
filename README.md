# 📈 AUTOFORECAST  

<br/>

Predicting the future has never been easier! **AUTOFORECAST** is an automated time series forecasting application that simplifies generating predictions using historical data.  
With support for various estimators, transformers, and metrics from `sktime`, AUTOFORECAST offers flexibility and precision in time series forecasting.  


##### 🚀 [Try AUTOFORECAST](https://autoforecast-bacfbzfudkg5fhbd.centralindia-01.azurewebsites.net/)
 
![img](https://imgs.search.brave.com/5UIXoDyhg78HrrBIsJIqOC7RdTIc_4aLNRdWfSWc8_E/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly90My5m/dGNkbi5uZXQvanBn/LzA5LzY4LzE0LzE4/LzM2MF9GXzk2ODE0/MTg4OV92Y2Z6QkxT/c1NLSDdyS2FkcjhL/aFBQbGxJYlZBWmxa/Yi5qcGc)

<br/>

## ⚠️ Important Notice  

**Please Note**: Due to limited computational resources, the deployed app may experience performance issues, or it may even crash if you select too many estimators (or if the dataset is large). The current instance has only **1.75 GB of RAM**, so to fully experience the capabilities of AUTOFORECAST, it is recommended to run the application locally.  

<br/>

## 🚀 Features  

- **Automated Time Series Forecasting**: Supports multiple estimators, transformers, and metrics from `sktime`.  
- **Customizable Pipelines**: Choose from 5 models, 5 transformers, and 5 metrics to create tailored forecasting solutions.  
- **Streamlined Visualization**: Plots to analyze forecasting results.  
- **Streamlit-based UI**: A user-friendly interface for seamless experimentation.  
- **ZenML Pipeline Orchestration**: Modular pipelines for scalable and reproducible forecasting workflows.  
- **Containerized Deployment**: Fully Dockerized for easy deployment on cloud or local environments.  

<br/>

## 🛠️ Tech Stack  

- **Time Series Analysis**: `sktime`  
- **Data Handling**: `NumPy`, `Pandas`  
- **Visualization**: `Matplotlib`, `sktime`  
- **Pipeline Orchestration**: `ZenML`  
- **User Interface**: `Streamlit`  
- **Containerization**: `Docker`  
- **Deployment**: `Azure Portal`

<br/>

## 🏗️ Project Structure  

```
AUTOFORECAST/  
│  
├── .github/workflows/        # GitHub Actions workflows  
├── src/  
│   └── AUTOFORECAST/  
│       ├── components/       # Pipeline stage modules  
│       ├── utils/            # Utility functions  
│       ├── config/           # Component configurations  
│       ├── pipeline/         # Pipeline stage scripts  
│       ├── entity/           # Data entity classes  
│       └── constants/        # Project constants  
├── config/                   # Global configuration files  
├── app.py                    # Streamlit application  
├── Dockerfile                # Docker configuration  
├── requirements.txt          # Python dependencies  
├── pyproject.toml            # Project metadata and build requirements  
├── params.yaml               # Pipeline parameters  
└── format.sh                 # Shell script for code formatting  
```

<br/>

## 🚀 Setup  

### Prerequisites  

- Python >= 3.8  
- pip  
- Docker (optional, for containerization)  

<br/>

### Installation  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/sanskarmodi8/AUTOFORECAST.git  
   cd AUTOFORECAST  
   ```  

2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```

<br/>

## 🖥️ Usage  

### Launching the Application Locally  

To run the application on your local machine:  
```bash  
streamlit run app.py  
```  

<br/>

## 🔮 Future Goals  

- [ ] Deploy on a powerful instance with atleast 16gb ram.  
- [ ] Add support for more of the sktime estimators.  
- [ ] Add support for multivariate forecasting.  
- [ ] Add support for forecasting with exogenous data.

<br/>

## 🤝 Contributing  

We welcome contributions to improve AUTOFORECAST!  

1. Fork the repository  
2. Create your feature branch: `git checkout -b feature/amazing-feature`  
3. Commit your changes: `git commit -am 'Add amazing feature'`  
4. Push to the branch: `git push origin feature/amazing-feature`  
5. Open a Pull Request  

<br/>

## 📄 License  

This project is licensed under the [MIT License](LICENSE).  
