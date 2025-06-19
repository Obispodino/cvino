# 🍷 CvalVino (Wine Recommendation System)

CvalVino is a comprehensive wine recommendation system that combines machine learning, computer vision, and wine expe## 📊 Project Structure

```bashse to help users discover wines based on their preferences, food pairings, or by analyzing wine bottle labels.

## 🚀 Features

- **Wine Recommendations**: Get personalized wine recommendations based on grape variety, wine type, body, ABV, region, and more
- **Food Pairing**: Find the perfect wine to complement your meal
- **Wine Label AI**: Take a photo of a wine label and get information about the wine and similar recommendations
- **Multiple Interfaces**:
  - Web application built with Streamlit
  - REST API powered by FastAPI

## 🛠️ Architecture

The system consists of several integrated components:

- **FastAPI Backend**: Provides RESTful endpoints for wine recommendations and label analysis
- **Wine Label Analysis**: Uses Anthropic's Claude AI to extract information from wine label images
- **Recommendation Engine**: Custom ML algorithms to match wines based on user preferences
- **Docker Support**: Easy containerization and deployment

## 🔧 Installation & Setup

### Prerequisites

- Python 3.11+
- Docker (optional)
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
# Create a .env file with required variables
# Make sure to add your Anthropic API key
```

### Running the Application

#### FastAPI Backend (Local)

```bash
uvicorn API.fast:app --reload
```

API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs)

## 🐳 Docker Deployment

Build and run the Docker container:

```bash
# Build Docker image
make docker_build_local

# Run Docker container
make docker_run_local
```

The API will be accessible at [http://localhost:8501](http://localhost:8501)

## 🔄 API Endpoints

### Wine Recommendations

**Endpoint**: `/recommend-wines`
**Method**: POST
**Description**: Get wine recommendations based on specific characteristics

**Example Request**:

```json
{
  "wine_type": "Red",
  "grape_varieties": ["Malbec"],
  "body": "Full-bodied",
  "abv": 14.5,
  "acidity": null,
  "country": null,
  "region_name": null,
  "n_recommendations": 5
}
```

### Food Pairing Recommendations

**Endpoint**: `/recommend-by-food`
**Method**: POST
**Description**: Get wine recommendations that pair well with specific foods

**Example Request**:

```json
{
  "food_pairing": "Chicken",
  "wine_type": "Red",
  "grape_varieties": ["Malbec"],
  "body": null,
  "abv": 14,
  "acidity": null,
  "country": null,
  "region_name": null,
  "n_recommendations": 5,
  "exact_match_only": false
}
```

### Wine Label Scanner

**Endpoint**: `/read_image`
**Method**: POST
**Description**: Upload a wine label image to extract information and get similar wine recommendations

**Parameters**:

- `img`: The image file (multipart/form-data)
- `n_recommendations`: Number of similar wines to recommend (optional)

**Example Response**:

```json
{
  "wine_type": "White",
  "grape_varieties": ["Alvarinho"],
  "body": "Light-bodied",
  "acidity": "High",
  "country": "Portugal",
  "region": "Vinho Verde",
  "ABV": "10.5",
  "extraction_successful": true,
  "recommendations": [
    {
      "WineID": 12345,
      "WineName": "Quinta do Ameal Loureiro Vinho Verde",
      "Type": "White",
      "Grapes": "Loureiro",
      "ABV": 10.0,
      "Body": "Light-bodied",
      "Acidity": "High",
      "Country": "Portugal",
      "RegionName": "Vinho Verde",
      "Similarity": 0.92
    }
  ]
}
```
http://localhost:8501/

## � Project Structure

```
cvino/
├── API/                  # FastAPI application
│   └── fast.py           # Main API endpoints
├── cv_functions/         # Core functionality
│   ├── encoder.py        # Feature encoding
│   ├── food_recommendation.py  # Food pairing logic
│   ├── geocode_regions.py  # Location handling
│   ├── model.py          # ML model operations
│   ├── recommendation.py  # Wine recommendation logic
│   └── wine_label_ai2.py  # Image analysis
├── models/               # Trained ML models
├── raw_data/             # Wine metadata
├── .dockerignore         # Docker exclusion patterns
├── .env                  # Environment variables
├── Dockerfile            # Docker configuration
├── Makefile              # Automation commands
└── requirements.txt      # Python dependencies
```

## ⚠️ Troubleshooting

### Common Issues

- **API Error with "None" values**: When making API requests, use `null` instead of `"None"` or `"string"` for empty values:

```json
{
  "acidity": null,
  "country": null
}
```

- **Docker Build Errors**: Make sure your environment variables are properly set in `.env` and that Docker has sufficient resources

- **Model Loading Errors**: Ensure all required data files and models are available in the appropriate directories

## 🌟 Current Status

The project is functional with all core features implemented. Recent updates include:

- Enhanced `/read_image` endpoint to return both wine information and similar recommendations
- Fixed handling of null values in API requests
- Improved error handling and user feedback
- Optimized Docker configuration
- Added comprehensive API documentation

## 📝 License

[License information goes here]

## 👥 Contributors

- Obispodino Dino
- Meng-Jung
- gaziza-jnb
- ClaudiaK191 Claudia Kettmann

---
Citations:

@Article{bdcc7010020,
AUTHOR = {de Azambuja, Rogério Xavier and Morais, A. Jorge and Filipe, Vítor},
TITLE = {X-Wines: A Wine Dataset for Recommender Systems and Machine Learning},
JOURNAL = {Big Data and Cognitive Computing},
VOLUME = {7},
YEAR = {2023},
NUMBER = {1},
ARTICLE-NUMBER = {20},
URL = {https://www.mdpi.com/2504-2289/7/1/20},
ISSN = {2504-2289},
ABSTRACT = {In the current technological scenario of artificial intelligence growth, especially using machine learning, large datasets are necessary. Recommender systems appear with increasing frequency with different techniques for information filtering. Few large wine datasets are available for use with wine recommender systems. This work presents X-Wines, a new and consistent wine dataset containing 100,000 instances and 21 million real evaluations carried out by users. Data were collected on the open Web in 2022 and pre-processed for wider free use. They refer to the scale 1&ndash;5 ratings carried out over a period of 10 years (2012&ndash;2021) for wines produced in 62 different countries. A demonstration of some applications using X-Wines in the scope of recommender systems with deep learning algorithms is also presented.},
DOI = {10.3390/bdcc7010020}
}

Developed with ❤️ by the CvalVino team
