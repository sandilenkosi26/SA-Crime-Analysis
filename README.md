# 🧠 Crime Hotspot Classification and Forecasting in South Africa


---

##  Project Overview
Crime in South Africa remains one of the country’s most pressing social and economic challenges.  
This project applies **Machine Learning (Classification and Time Series Forecasting)** to identify crime hotspots and predict future crime trends.  
The goal is to turn raw SAPS statistics into actionable intelligence that can guide **law enforcement** and **policy decisions**.  

Additionally, a **Streamlit Dashboard** was developed to make the analysis interactive and easy to interpret.  
Finally, a **Drone Simulation** was designed to illustrate how autonomous systems could monitor high-crime areas efficiently.  

---

## Datasets

| Dataset Name | Source | Description | Purpose |
|---------------|---------|-------------|----------|
| `aggravated_robbery_incidents_by_category.csv` | SAPS Open Data | Records the number of aggravated robbery incidents per province and financial year. | Used for classification and forecasting models. |
| `ProvincePopulation.csv` | Statistics South Africa (Stats SA) | Provides population, area, and density per province. | Provides contextual socio-economic data and supports the multi-relational merge. |

### Dataset Justification
- **Relevance:** Both datasets directly support the tasks of classification (crime hotspots) and forecasting (crime trends).  
- **Completeness:** Data spans multiple years and covers all provinces in South Africa.  
- **Credibility:** Both datasets are sourced from **official South African government** data providers.  
- **Limitations:** Some datasets may have underreporting or differences in data collection across years.

---

##  Data Cleaning and Preparation
- Columns were standardized (lowercased, stripped spaces).  
- Missing values were handled and unnecessary columns removed.  
- Datasets were merged using the common field `province`.  
- A new column `incident_count` was created from the `count` field for consistency.  
- A binary column `Hotspot` was added, where the top 25% of provinces (based on incident counts) were classified as hotspots.

---

##  Exploratory Data Analysis (EDA)
The EDA revealed key insights about aggravated robbery patterns:

- **Bar Chart:** Compared total incidents by province.  
- **Pie Chart:** Showed the share of total incidents per province.  
- **Histogram:** Displayed the distribution of crime counts.  

Provinces such as **Gauteng** and **KwaZulu-Natal** emerged as consistent high-crime areas.

---

##  Classification Model (Crime Hotspots)
- **Model:** Decision Tree Classifier  
- **Goal:** Predict whether a province qualifies as a **hotspot (1)** or **non-hotspot (0)**  
- **Features Used:** `population`, `density`, and `incident_count`  
- **Evaluation Metrics:** Accuracy Score, Confusion Matrix, Classification Report  
- **Results:** Model achieved strong accuracy, correctly identifying high-risk areas.  

This classification can help SAPS allocate patrols and resources to the most critical areas.

---

##  Forecasting Model (Crime Trends)
- **Model Used:** Linear Regression  
- **Goal:** Forecast aggravated robbery incidents for the next 5 years  
- **Results:** Forecast indicates a potential upward trend in robbery cases, suggesting a need for preventive strategies.  
- **Visualization:** Combined actual and predicted values with confidence intervals.

---

##  Streamlit Dashboard
An interactive dashboard (`app.py`) was built using **Streamlit** to display:

- EDA visualizations (bar, pie, histogram)  
- Classification results with confusion matrix  
- Time series forecasts with interactive filtering  
- Text summaries for both technical and non-technical audiences  

### Run the App:
```bash
streamlit run app.py
