import streamlit as st
import pandas as pd
import os

# Set up page
st.set_page_config(page_title="🍷 CVINO", layout="wide")
st.title("🍷CVINO")

# === Load the dataset ===
@st.cache_data
def load_data():
    file_path = os.path.join("raw_data", "last", "XWines_Full_100K_wines.csv")
    return pd.read_csv(file_path)

df = load_data()

# === User input ===
st.subheader("🔎 Enter your wine preferences")
col1, col2 = st.columns(2)

# Grape input
with col1:
    grape_input = st.text_input("Grape", placeholder="e.g., Merlot")

# Wine type dropdown
with col2:
    wine_type_input = st.selectbox(
        "Type",
        df["Type"].dropna().unique().tolist()
    )

# Create 2 more columns
col3, col4 = st.columns(2)

# Body preference
with col3:
    body_input = st.selectbox(
        "Body",
        ["Very light-bodied", "Light-bodied", "Medium-bodied", "Full-bodied", "Very full-bodied"]
    )

# ABV range slider
with col4:
    abv_input = st.slider(
        "ABV (%)",
        min_value=5.0,
        max_value=20.0,
        value=12.5,  # Default value
        step=0.1
    )

# Number of recommendations (below filters)
num_recommendations = st.slider("Number of Recommendations", 1, 20, 5)

# === Filter logic ===
filtered_df = df.copy()

if grape_input:
    filtered_df = filtered_df[filtered_df["Grapes"].str.contains(grape_input, case=False, na=False)]

filtered_df = filtered_df[filtered_df["Body"] == body_input]

filtered_df = filtered_df[
    (filtered_df["ABV"] >= abv_input - 1) & (filtered_df["ABV"] <= abv_input + 1)
]

# === Display results as cards ===
st.subheader("🍇 Recommended Wines")

if filtered_df.empty:
    st.warning("No matching wines found. Try changing the filters.")
else:
    top_wines = filtered_df[["WineName", "Grapes", "Body", "ABV", "RegionName", "Country"]].head(10)

    for index, row in top_wines.iterrows():
        with st.container():
            cols = st.columns([1, 4])
            with cols[0]:
                st.image("https://cdn-icons-png.flaticon.com/512/3523/3523063.png", width=80) # Wine icon or placeholder
            with cols[1]:
                st.markdown(f"""
                    ### {row['WineName']}
                    - **Grapes**: {row['Grapes']}
                    - **Body**: {row['Body']}
                    - **ABV**: {row['ABV']}%
                    - **Region**: {row['RegionName']}
                    - **Country**: {row['Country']}
                    """)
                st.markdown("---")
