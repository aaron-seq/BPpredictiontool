# Advanced Blood Pressure Prediction Tool 2.1

[![Build Status](https://github.com/aaronseq12/BPpredictiontool/workflows/Build%20and%20Deploy/badge.svg)](https://github.com/aaronseq12/BPpredictiontool/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/react-18.0+-61dafb.svg)](https://reactjs.org/)

## 🌟 Overview

An advanced AI-powered web application that predicts **Arterial Blood Pressure (ABP)** from **Photoplethysmography (PPG)** and **Electrocardiogram (ECG)** signals using state-of-the-art deep learning techniques. The tool features real-time signal quality assessment, interactive visualizations, and a modern responsive web interface.

### ✨ Key Features

- 🧠 **Advanced LSTM with Attention Mechanism** - Enhanced prediction accuracy
- 📊 **Real-time Signal Quality Assessment** - Validates input signal quality
- 🎨 **Modern Interactive UI** - Responsive design with smooth animations
- 📈 **Comprehensive Visualizations** - Interactive charts and real-time plotting
- 🔄 **Advanced Signal Processing** - Noise reduction and filtering algorithms
- 🚀 **Production-Ready Architecture** - Docker containers, CI/CD, monitoring
- 📱 **Multi-platform Support** - Works on desktop, tablet, and mobile
- 🔒 **Secure API** - Input validation, error handling, and structured logging

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │   ML Models     │
│   (React 18)    │◄──►│   (Flask API)    │◄──►│   (TensorFlow)  │
│                 │    │                  │    │                 │
│ • Modern UI     │    │ • RESTful API    │    │ • LSTM + Attn   │
│ • Visualizations│    │ • Validation     │    │ • Signal Proc   │
│ • Animations    │    │ • Error Handling │    │ • Quality Assess│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Git
- Docker (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/aaronseq12/BPpredictiontool.git
cd BPpredictiontool
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create data directory
mkdir data models logs

# Download dataset (see Dataset section below)
# Place .mat files in the data directory

# Train the model
python train_advanced_model.py

# Start the API server
python main.py
```

The backend will be available at `http://localhost:5000`

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create environment file
echo "REACT_APP_API_URL=http://localhost:5000" > .env

# Start development server
npm start
```

The frontend will be available at `http://localhost:3000`

### 4. Docker Setup (Recommended)

```bash
# Build and run all services
docker-compose up --build

# For production deployment
docker-compose --profile production up
```

## 📊 Dataset

The model uses the **Blood Pressure Dataset** from Kaggle:

1. Download from: [https://www.kaggle.com/mkachuee/BloodPressureDataset](https://www.kaggle.com/mkachuee/BloodPressureDataset)
2. Extract `.mat` files to `backend/data/` directory
3. The dataset contains synchronized PPG, ECG, and ABP signals from multiple subjects

### Data Structure
- **PPG signals**: Photoplethysmography recordings
- **ECG signals**: Electrocardiogram recordings  
- **ABP signals**: Arterial Blood Pressure (ground truth)
- **Sample rate**: 125 Hz
- **Sequence length**: 1000 samples (8 seconds)

## 🧠 Model Architecture

### Advanced LSTM with Attention

```python
Input(1000, 2) → PPG + ECG signals
    ↓
Multi-layer LSTM (128, 64, 32 units)
    ↓
Multi-Head Attention (8 heads)
    ↓
Residual Connections + Layer Normalization
    ↓
TimeDistributed Dense Layers
    ↓
Output(1000, 1) → Predicted ABP waveform
```

### Key Innovations

- **Multi-Head Attention**: Captures temporal dependencies
- **Residual Connections**: Improves gradient flow
- **Layer Normalization**: Stabilizes training
- **Signal Quality Assessment**: Pre-processing validation
- **Advanced Filtering**: Noise reduction and artifact removal

## 🎯 Performance Metrics

| Metric | Value |
|--------|-------|
| **Mean Absolute Error (MAE)** | < 5 mmHg |
| **Root Mean Square Error (RMSE)** | < 8 mmHg |
| **Correlation Coefficient** | > 0.85 |
| **Prediction Accuracy (±5mmHg)** | > 90% |

## 🔧 API Documentation

### Health Check
```bash
GET /health
```

### Predict Blood Pressure
```bash
POST /predict
Content-Type: application/json

{
  "ppg_signal": [array of 1000 float values],
  "ecg_signal": [array of 1000 float values],
  "include_quality_metrics": true
}
```

### Response
```json
{
  "predicted_abp": [array of 1000 predicted values],
  "systolic_bp": 120.5,
  "diastolic_bp": 80.2,
  "mean_arterial_pressure": 93.6,
  "confidence_score": 0.87,
  "quality_metrics": {
    "ppg_quality": {...},
    "ecg_quality": {...},
    "overall_score": 0.87
  }
}
```

## 🎨 Frontend Features

### Modern UI Components
- **Interactive Signal Visualization** - Real-time plotting with Chart.js
- **Signal Quality Dashboard** - Live quality assessment
- **Responsive Design** - Optimized for all screen sizes
- **Smooth Animations** - Framer Motion animations
- **Dark/Light Mode** - Theme switching capability
- **Progress Indicators** - Real-time processing feedback

### Sample Data Generator
The tool includes built-in sample data generators for:
- Normal blood pressure patterns
- Hypertensive patterns  
- Hypotensive patterns
- Arrhythmic patterns

## 🚀 Deployment

### Backend Deployment (Railway)

1. **Connect Repository**
   ```bash
   # Railway will automatically detect the Dockerfile
   railway login
   railway link [project-id]
   railway up
   ```

2. **Environment Variables**
   ```
   DEBUG=False
   HOST=0.0.0.0
   PORT=5000
   CORS_ORIGINS=https://your-frontend-domain.com
   ```

### Frontend Deployment (Vercel)

1. **Connect Repository to Vercel**
2. **Build Settings**
   ```
   Build Command: npm run build:production
   Output Directory: build
   Install Command: npm ci
   ```

3. **Environment Variables**
   ```
   REACT_APP_API_URL=https://your-backend-url.railway.app
   ```

### Alternative Deployments

- **Heroku**: Use `Procfile` for backend deployment
- **AWS**: Deploy using EC2, ECS, or Lambda
- **Google Cloud**: Use Cloud Run or App Engine
- **Azure**: Deploy with App Service

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest --cov=./ --cov-report=html
```

### Frontend Tests
```bash
cd frontend
npm test -- --coverage --watchAll=false
```

### Integration Tests
```bash
# Run both backend and frontend
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## 📁 Project Structure

```
BPpredictiontool/
├── 📁 backend/                  # Python Flask API
│   ├── 📁 api/                 # API routes and validation
│   ├── 📁 data_processing/     # Data preprocessing modules
│   ├── 📁 models/              # ML model implementations
│   ├── 📁 utils/               # Utility functions
│   ├── 📄 config.py            # Configuration settings
│   ├── 📄 main.py              # Flask application entry
│   └── 📄 requirements.txt     # Python dependencies
├── 📁 frontend/                # React web application
│   ├── 📁 public/              # Static assets
│   ├── 📁 src/                 # React source code
│   │   ├── 📁 components/      # React components
│   │   ├── 📁 hooks/           # Custom React hooks
│   │   ├── 📁 utils/           # Utility functions
│   │   └── 📁 styles/          # CSS and styling
│   └── 📄 package.json         # Node.js dependencies
├── 📁 .github/                 # GitHub Actions CI/CD
├── 📄 docker-compose.yml       # Docker orchestration
├── 📄 Dockerfile               # Docker container config
└── 📄 README.md               # This file
```

## 🛠️ Development

### Setting up Development Environment

1. **Install development dependencies**
   ```bash
   # Backend
   cd backend
   pip install -r requirements.txt
   pip install pytest black flake8 mypy

   # Frontend  
   cd frontend
   npm install
   npm install --save-dev @testing-library/react
   ```

2. **Pre-commit hooks**
   ```bash
   pre-commit install
   ```

3. **Code formatting**
   ```bash
   # Backend
   black . && flake8 .

   # Frontend
   npm run lint && npm run format
   ```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes with tests
3. Run test suite: `npm test` or `pytest`
4. Submit pull request

## 🔒 Security Considerations

- **Input validation** on all API endpoints
- **CORS protection** with configurable origins
- **Rate limiting** to prevent abuse
- **Secure headers** in HTTP responses
- **Environment variable** management for secrets
- **Container security** with non-root users

## 📈 Performance Optimization

- **Model quantization** for faster inference
- **Request caching** for repeated predictions
- **Lazy loading** of frontend components
- **Image optimization** and compression
- **CDN integration** for static assets
- **Database connection pooling**

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Write tests for new features
- Follow code style guidelines (Black for Python, Prettier for JavaScript)
- Update documentation for new features
- Ensure all CI checks pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- **Dataset**: Thanks to the creators of the Blood Pressure Dataset on Kaggle
- **TensorFlow Team**: For the excellent deep learning framework
- **React Community**: For the amazing frontend ecosystem
- **Open Source Contributors**: For various libraries and tools used

## 📚 Scientific References

1. Kurylyak, Y., Lamonaca, F., & Grimaldi, D. (2013). A Neural Network-based method for continuous blood pressure estimation from a PPG signal. *IEEE International Instrumentation and Measurement Technology Conference*.

2. Liang, Y., Chen, Z., Ward, R., & Elgendi, M. (2018). Hypertension assessment via ECG and PPG signals: An evaluation using MIMIC database. *Diagnostics*, 8(3), 65.

3. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. *ICLR 2015*.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/aaronseq12/BPpredictiontool/issues)
- **Discussions**: [GitHub Discussions](https://github.com/aaronseq12/BPpredictiontool/discussions)
- **Email**: aaronsequeira12@gmail.com

---

**⭐ Star this repository if you found it helpful!**

*Made with ❤️ by [Aaron Sequeira](https://github.com/aaronseq12)*
