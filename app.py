import streamlit as st
import pandas as pd
import os
import ast

# === Set up page ===
st.set_page_config(page_title="🍇 CvalVino", layout="wide")

# === Add custom styling ===
st.markdown(
    """
    <style>
    /* Set elegant wine background */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #722F37;
        color: #FFFFFF;
    }

    h1, h2, h3, h4 {
        color: #FFFFFF;
        font-family: 'Georgia', serif;
    }

    .stContainer {
        background-color: #F4A6B1;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }

    /* Force black style specifically on the food button */
    button:has(span:contains("Get Wine Recommendation by Food")) {
        background-color: #000 !important;
        color: white !important;
        font-weight: bold;
        border: none !important;
        box-shadow: none !important;
    }

    button:has(span:contains("Get Wine Recommendation by Food")):hover {
        background-color: #222 !important;
        color: white !important;
    }

    /* Force all markdown and label text to white */
    label,
    .css-1aumxhk,
    .css-1cpxqw2,
    .css-1v0mbdj,
    .css-1d391kg,
    .css-10trblm,
    .css-qrbaxs,
    .css-1v3fvcr,
    .css-ffhzg2,
    .e1nzilvr5,
    .e1nzilvr1 {
        color: #FFFFFF !important;
    }

    label[data-testid="stSelectboxLabel"] {
        color: #FFFFFF !important;
        font-weight: bold;
    }

    /* Make input/select boxes white */
    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input,
    .stSlider,
    .stSlider > div {
        background-color: white !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# === Load the dataset ===
@st.cache_data
def load_data():
    file_path = os.path.join("raw_data", "last", "XWines_Full_100K_wines.csv")
    return pd.read_csv(file_path)

df = load_data()

# === Session state for page navigation ===
if "food_page" not in st.session_state:
    st.session_state.food_page = False

# === Extract unique foods from Harmonize column ===
@st.cache_data
def get_unique_foods(df):
    food_set = set()
    for item in df["Harmonize"].dropna():
        try:
            foods = ast.literal_eval(item)
            food_set.update([f.strip() for f in foods])
        except:
            continue
    return sorted(food_set)

unique_foods = get_unique_foods(df)

# === Title and navigation ===
st.title(":wine_glass: CvalVino")

if not st.session_state.food_page:
    if st.button("🍽️ Get Wine Recommendation by Food"):
        st.session_state.food_page = True

# === Food Recommendation Page ===
if st.session_state.food_page:
    st.header("🍽️ Food-Based Wine Recommendation")

    food_input = st.selectbox("Choose a food to get wine recommendations:", unique_foods, key="food_input")

    if st.button("🔎 Recommend Wines"):
        if food_input.strip():
            food_wines = df[df["Harmonize"].apply(
                lambda x: food_input.lower() in [item.lower() for item in ast.literal_eval(x)] if isinstance(x, str) and x.startswith("[") else False
            )]

            if not food_wines.empty:
                st.success(f"Found {len(food_wines)} wines for '{food_input}' 🍇")
                top_food_wines = food_wines[["WineName", "Grapes", "Body", "ABV", "RegionName", "Country", "Harmonize"]].head(10)

                for _, row in top_food_wines.iterrows():
                    with st.container():
                        cols = st.columns([1, 4])
                        with cols[0]:
                            st.image("https://purepng.com/public/uploads/large/purepng.com-wine-bottlefood-winebottlealcoholbeverageliquor-2515194557124w46mz.png", width=80)
                        with cols[1]:
                            st.markdown(f"""
                                ### {row['WineName']}
                                - **Grapes**: {", ".join(ast.literal_eval(row['Grapes'])) if isinstance(row['Grapes'], str) and row['Grapes'].startswith("[") else row['Grapes']}
                                - **Body**: {row['Body']}
                                - **ABV**: {row['ABV']}%
                                - **Region**: {row['RegionName']}
                                - **Country**: {row['Country']}
                                - **Food Recommendation**: {", ".join(ast.literal_eval(row['Harmonize']))}
                            """)
                            st.markdown("---")
            else:
                st.warning(f"No wine recommendations found for '{food_input}'.")
        else:
            st.warning("Please enter a food.")

    if st.button("🔙 Back to Main Page"):
        st.session_state.food_page = False
        del st.session_state["food_input"]
        st.rerun()

# === Main Wine Filters Page ===
if not st.session_state.food_page:
    st.subheader("🔎 Enter your wine preferences")

    col1, col2 = st.columns(2)

    with col1:
        grape_input = st.text_input("Grape", placeholder="e.g., Merlot, Cabernet Sauvignon")
        wine_name_input = None
        if grape_input:
            filtered_wines = df[df["Grapes"].str.contains(grape_input.strip(), case=False, na=False)]
            wine_name_options = filtered_wines["WineName"].dropna().unique().tolist()

            if wine_name_options:
                wine_name_input = st.selectbox("Wine Name (filtered by grape)", wine_name_options)
            else:
                st.warning("No wines found for that grape.")

        country_input = st.selectbox("Country", df["Country"].dropna().unique().tolist())

    with col2:
        wine_type_input = st.selectbox("Type", df["Type"].dropna().unique().tolist())
        region_input = st.selectbox("Region", df["RegionName"].dropna().unique().tolist())

    col3, col4 = st.columns(2)
    with col3:
        body_input = st.selectbox("Body", ["Very light-bodied", "Light-bodied", "Medium-bodied", "Full-bodied", "Very full-bodied"])
    with col4:
        abv_input = st.slider("ABV (%)", min_value=5.0, max_value=20.0, value=12.5, step=0.1)

    num_recommendations = st.number_input("How many wine recommendations would you like to see?", min_value=1, max_value=50, value=10, step=1)

    # === Filter logic ===
    filtered_df = df.copy()
    if grape_input:
        filtered_df = filtered_df[filtered_df["Grapes"].str.contains(grape_input, case=False, na=False)]
    if wine_name_input:
        filtered_df = filtered_df[filtered_df["WineName"] == wine_name_input]

    filtered_df = filtered_df[filtered_df["Body"] == body_input]
    filtered_df = filtered_df[(filtered_df["ABV"] >= abv_input - 1) & (filtered_df["ABV"] <= abv_input + 1)]
    filtered_df = filtered_df[filtered_df["RegionName"] == region_input]

    # === Display results as cards ===
    st.subheader("🍇 Recommended Wines")

    if filtered_df.empty:
        st.warning("No matching wines found. Try changing the filters.")
    else:
        top_wines = filtered_df[["WineName", "Grapes", "Body", "ABV", "RegionName", "Country", "Harmonize"]].head(num_recommendations)

        for index, row in top_wines.iterrows():
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    st.image("https://purepng.com/public/uploads/large/purepng.com-wine-bottlefood-winebottlealcoholbeverageliquor-2515194557124w46mz.png", width=80)
                with cols[1]:
                    st.markdown(f"""
                        ### {row['WineName']}
                        - **Grapes**: {", ".join(ast.literal_eval(row['Grapes'])) if isinstance(row['Grapes'], str) and row['Grapes'].startswith("[") else row['Grapes']}
                        - **Body**: {row['Body']}
                        - **ABV**: {row['ABV']}%
                        - **Region**: {row['RegionName']}
                        - **Country**: {row['Country']}
                        - **Food Recommendation**: {", ".join(ast.literal_eval(row["Harmonize"]))}
                    """)
                    st.markdown("---")
