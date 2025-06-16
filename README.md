# 🍷 CvalVino (Wine Recommendation System)

CvalVino is a comprehensive wine recommendation system that combines machine learning, computer vision, and wine expertise to help users discover wines based on their preferences, food pairings, or even by analyzing wine bottle labels.

## 🚀 Features

- **Wine Recommendations**: Get personalized wine recommendations based on grape variety, wine type, body, ABV, region, and more
- **Food Pairing**: Find the perfect wine to complement your meal
- **Wine Label AI**: Take a photo of a wine label and get information about the wine and similar recommendations
- **Multiple Interfaces**:
  - Web application built with Streamlit
  - REST API powered by FastAPI

## 🛠️ Architecture

The system consists of several integrated components:

- **Streamlit Interface**: User-friendly web application for interacting with the recommendation system
- **FastAPI Backend**: Provides RESTful endpoints for wine recommendations and label analysis
- **Wine Label Analysis**: Uses Anthropic's Claude AI to extract information from wine label images
- **Recommendation Engine**: Custom ML algorithms to match wines based on user preferences
- **Data Pipeline**: Preprocessing, feature engineering, and model training components

## 🔧 Installation & Setup

### Prerequisites

- Python 3.11+
- Anthropic API key (for wine label recognition)

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/cvino.git
cd cvino
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env.yaml file based on the sample
cp .env.yaml.sample .env.yaml
# Edit the file to add your Anthropic API key
```

### Running the Application

#### Streamlit Web Interface

```bash
streamlit run app.py
```

Access the web interface at http://localhost:8501

#### FastAPI Backend

```bash
uvicorn API.fast:app --reload
```

API documentation will be available at http://localhost:8000/docs

## 🐳 Docker Deployment

Build and run the Docker container:

```bash
docker build -t cvino .
docker run -p 8080:8080 -e ANTHROPIC_API_KEY=your-key-here cvino
```

## 📊 Data

The system uses a comprehensive dataset of wines with various attributes:
- Wine names, types, and grape varieties
- Body, acidity, and ABV information
- Regions and countries of origin
- Food pairing recommendations

## 🧠 Machine Learning Components

- **Custom Encoders**: Specialized transformers for wine data
- **Recommendation Model**: Similarity-based recommendation system
- **Wine Label AI**: Computer vision powered by Claude API to extract information from wine labels

## 🔄 API Endpoints

### Wine Recommendations

- `POST /recommendations`: Get wine recommendations based on specified characteristics
- `POST /analyze-label`: Extract information from a wine label image
- `POST /recommendations-from-label`: Get wine recommendations based on a wine label image

## 📝 License

[License information goes here]

## 👥 Contributors

[List of contributors goes here]

---

Developed with ❤️ by the CvalVino team
