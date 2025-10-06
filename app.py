import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('merged_dataset.csv')  # <- change if your merged dataset has a different name
    return df

df = load_data()

st.title("📊 South Africa Crime Analysis Dashboard")
st.markdown("This dashboard presents **Crime Hotspot Classification** and **Crime Forecasting** using machine learning models. Developed by *Sandile Nkosi* for the Technical Programming 2 Final Exam.")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Data")
selected_province = st.sidebar.selectbox("Select Province", df['province'].unique())
selected_year = st.sidebar.selectbox("Select Financial Year", df['financial_year'].unique())

filtered_df = df[(df['province'] == selected_province) & (df['financial_year'] == selected_year)]

# --- EDA Section ---
st.header("📊 Exploratory Data Analysis (EDA)")

# Bar Chart
st.subheader("Incidents by Province")
fig1, ax1 = plt.subplots(figsize=(10,6))
sns.barplot(data=df, x='province', y='incident_count', palette='viridis', ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# Pie Chart
st.subheader("Proportion of Incidents by Province")
fig2, ax2 = plt.subplots(figsize=(8,8))
df.groupby('province')['count'].sum().plot.pie(autopct='%1.1f%%', startangle=120, ax=ax2)
plt.ylabel('')
st.pyplot(fig2)

# --- Classification Section ---
st.header("🚨 Crime Hotspot Classification")

# Prepare Data
X = df[['population', 'density']]
y = df['Hotspot']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"**Model Accuracy:** {accuracy:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig3, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
st.pyplot(fig3)

st.markdown("Provinces classified as **Hotspots (1)** indicate high-crime areas requiring additional resources or patrols.")

# --- Forecasting Section ---
st.header("📈 Crime Forecasting")

ts_df = df.groupby('financial_year')['count'].sum().reset_index()
ts_df['year_numeric'] = ts_df['financial_year'].astype(str).str[:4].astype(int)

X = ts_df[['year_numeric']]
y = ts_df['count']
lr = LinearRegression()
lr.fit(X, y)

future_years = np.arange(ts_df['year_numeric'].max() + 1, ts_df['year_numeric'].max() + 6)
future_preds = lr.predict(future_years.reshape(-1, 1))

forecast_df = pd.DataFrame({'Year': future_years, 'Predicted Incidents': future_preds})

fig4, ax4 = plt.subplots(figsize=(10,6))
plt.plot(ts_df['year_numeric'], ts_df['count'], marker='o', label='Actual')
plt.plot(forecast_df['Year'], forecast_df['Predicted Incidents'], marker='x', linestyle='--', color='red', label='Forecast')
plt.title('Aggravated Robbery Forecast (Next 5 Years)')
plt.xlabel('Year')
plt.ylabel('Number of Incidents')
plt.legend()
plt.grid(True)
st.pyplot(fig4)

st.markdown("The forecast shows projected crime levels for the next 5 years, enabling early planning and resource allocation.")
