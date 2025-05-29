# 🎯 Advanced Twitter Sentiment Analysis & Real-Time Dashboard

> A comprehensive sentiment analysis platform with real-time Twitter monitoring, interactive visualizations, and production-ready ML models.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

- **Real-time Twitter Stream Processing** with live sentiment monitoring
- **Interactive Web Dashboard** built with Streamlit/Dash
- **Multiple ML Models** (Naive Bayes, LSTM, BERT, RoBERTa)
- **Advanced Text Preprocessing Pipeline** 
- **Comprehensive Model Evaluation** with detailed metrics
- **Data Visualization** and sentiment trend analysis
- **RESTful API** for model predictions
- **Docker Containerization** for easy deployment
- **CI/CD Pipeline** with automated testing




## 🛠️ Installation & Setup

### Quick Start with Docker
```bash
git clone https://github.com/yourusername/advanced-sentiment-analysis.git
cd advanced-sentiment-analysis
docker-compose up -d
```

### Manual Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/advanced-sentiment-analysis.git
cd advanced-sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('all')"

# Set up environment variables
cp .env.example .env
# Edit .env with your Twitter API credentials
```

## 📖 Usage

### 1. Data Preprocessing
```python
from src.data.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()
cleaned_text = preprocessor.clean_text("Your tweet text here! 😊 #sentiment")
```

### 2. Model Training
```bash
# Train all models
python src/models/train_models.py --config configs/training_config.yaml

# Train specific model
python src/models/train_models.py --model bert --epochs 10
```

### 3. Real-time Prediction
```python
from src.models.ensemble import SentimentEnsemble

model = SentimentEnsemble.load('models/ensemble_model.pkl')
sentiment = model.predict("I love this product!")
# Output: {'sentiment': 'positive', 'confidence': 0.94}
```

### 4. Launch Dashboard
```bash
streamlit run src/visualization/dashboard.py
```

### 5. API Server
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

## 🎯 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 0.82 | 0.81 | 0.82 | 0.81 |
| LSTM | 0.87 | 0.86 | 0.87 | 0.86 |
| BERT | 0.91 | 0.90 | 0.91 | 0.90 |
| RoBERTa | 0.93 | 0.92 | 0.93 | 0.92 |
| **Ensemble** | **0.94** | **0.93** | **0.94** | **0.94** |

## 📊 Dataset Information

- **Primary Dataset**: Sentiment140 (1.6M tweets)
- **Additional Sources**: 
  - Stanford Twitter Sentiment Dataset
  - Custom collected tweets via Twitter API
- **Preprocessing Steps**: 11-stage pipeline including emoji handling, spelling correction, and advanced tokenization

## 🔧 Advanced Features

### Text Preprocessing Pipeline
- Smart emoji detection and conversion
- Context-aware spell checking
- Advanced tokenization with custom rules
- Intelligent stop word removal
- Morphological analysis

### Model Ensemble
- Weighted voting based on model confidence
- Dynamic model selection based on text characteristics
- Real-time model performance monitoring

### Real-time Analytics
- Live Twitter stream processing
- Trend detection and anomaly identification
- Sentiment distribution tracking
- Geographic sentiment mapping

## 📈 Visualizations

The dashboard includes:
- **Real-time Sentiment Trends** - Live updating charts
- **Word Clouds** - Most common positive/negative terms
- **Geographic Heat Maps** - Sentiment by location
- **Model Comparison** - Performance metrics visualization
- **Confusion Matrices** - Detailed classification results

## 🔗 API Endpoints

```
POST /predict - Single text prediction
POST /batch_predict - Batch predictions
GET /models - Available models info
GET /health - Service health check
WebSocket /stream - Real-time predictions
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_models.py -v
pytest tests/test_api.py -v

# Generate coverage report
pytest --cov=src tests/
```

## 📊 Monitoring & Logging

- **Model Performance Monitoring** with MLflow
- **API Monitoring** with Prometheus metrics
- **Structured Logging** with custom formatters
- **Error Tracking** with detailed stack traces

## 🚀 Deployment

### Heroku Deployment
```bash
heroku create your-sentiment-app
git push heroku main
```

### AWS/GCP Deployment
See `docs/deployment.md` for detailed cloud deployment instructions.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📋 TODO

- [ ] Add support for multi-language sentiment analysis
- [ ] Implement active learning for model improvement
- [ ] Add more social media platforms (Reddit, Instagram)
- [ ] Create mobile app interface
- [ ] Add A/B testing framework for models

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Model Architecture](docs/models.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@ktkanas](https://github.com/ktkanas)
- LinkedIn: [Muhammad Anas](https://linkedin.com/in/anascs)
- Email: anasktk.official@gmail.com

## 🙏 Acknowledgments

- Twitter API for data access
- Hugging Face for transformer models
- The open-source community for amazing libraries

---

⭐ If you found this project helpful, please give it a star!
