import streamlit as st
import pandas as pd
import os
import ast
import requests

# === Set up page ===
st.set_page_config(page_title="🍇 CvalVino", layout="wide")

# === Styling ===
st.markdown("""<style>
html, body, [data-testid="stAppViewContainer"] { background-color: #722F37; color: #FFFFFF; }
h1, h2, h3, h4 { color: #FFFFFF; font-family: 'Georgia', serif; }
.stContainer { background-color: #F4A6B1; padding: 15px; border-radius: 15px; box-shadow: 2px 2px 10px rgba(0,0,0,0.4); margin-bottom: 20px; }
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {
    background-color: white !important; color: black !important;
}
.stSlider > div, .stSlider span, label, .css-1aumxhk, .css-1cpxqw2, .css-1v0mbdj, .css-1d391kg, .css-10trblm, .css-qrbaxs,
.css-1v3fvcr, .css-ffhzg2, .e1nzilvr5, .e1nzilvr1 {
    color: #FFFFFF !important;
}
label[data-testid="stSelectboxLabel"] {
    color: #FFFFFF !important; font-weight: bold;
}
.stButton button {

    background-color: #c24f5d !important;
    color: #FFFFFF !important;
    font-weight: bold;
    border: none;
    border-radius: 5px;
    padding: 20px 20px;

    transition: all 0.3s ease;
    width: 100% !important;
    min-width: 180px;
    max-width: 100%;
}
.stButton button:hover {

    background-color: #eb919c !important;
    color: #FFFFFF !important;
    transform: scale(1.05);

}
</style>""", unsafe_allow_html=True)

# === Title ===
st.markdown("""
<h1 style='text-align: center; color: white; font-family: Georgia, serif; font-size: 60px;'>🍇🍷 <b>CvalVino</b> 🍷🍇</h1>
<p style='text-align: center; color: white; font-family: Georgia, serif;'>Your Elegant Wine Recommender</p>
""", unsafe_allow_html=True)

# === Load dropdown source data ===
@st.cache_data
def load_data():
    try:
        return pd.read_csv("raw_data/wine_metadata.csv")
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

# create a Country vs Region lookup dictionary

country_to_regions = (
    df.groupby('Country')['RegionName']
    .apply(lambda x: list(dict.fromkeys(x)))  # removes duplicates, keeps order
    .to_dict()
)

region_to_country = {
    region: country
    for country, regions in country_to_regions.items()
    for region in regions}


# === Session state for page navigation ===
if "food_page" not in st.session_state:
    st.session_state.food_page = False

if "wine_page" not in st.session_state:
    st.session_state.wine_page = True


@st.cache_data
def get_unique_foods(df):
    if "Harmonize" not in df.columns:
        return []
    food_set = set()
    for item in df["Harmonize"].dropna():
        try:
            foods = ast.literal_eval(item)
            food_set.update([f.strip() for f in foods])
        except:
            continue
    return sorted(food_set)

# === Food-based Recommendation Section (Only if 'Harmonize' exists) ===
if "Harmonize" in df.columns:
    unique_foods = get_unique_foods(df)


# === create  buttons ===
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🍽️ Get Wine Recommendation by Food"):
        st.session_state.food_page = True
        st.session_state.wine_page = False

with col2:
    if st.button("🍷 Get Wine Recommendation by Characteristics"):
        st.session_state.wine_page = True
        st.session_state.food_page = False

# === Food-Based Wine Recommendation Page ===
if st.session_state.food_page:
    st.header("🍽️ Food-Based Wine Recommendation")

    food_emoji_dict = {
        'Aperitif': '🍹',
        'Appetizer': '🥟',
        'Asian Food': '🍜',
        'Baked Potato': '🥔',
        'Barbecue': '🍖',
        'Beans': '🫘',
        'Beef': '🥩',
        'Blue Cheese': '🧀',
        'Cake': '🍰',
        'Cheese': '🧀',
        'Chestnut': '🌰',
        'Chicken': '🍗',
        'Chocolate': '🍫',
        'Citric Dessert': '🍋',
        'Codfish': '🐟',
        'Cold Cuts': '🥓',
        'Cookies': '🍪',
        'Cream': '🥛',
        'Cured Meat': '🍖',
        'Curry Chicken': '🍛',
        'Dessert': '🍮',
        'Dried Fruits': '🍇',
        'Duck': '🦆',
        'Eggplant Parmigiana': '🧀',
        'Fish': '🐟',
        'French Fries': '🍟',
        'Fruit': '🍓',
        'Fruit Dessert': '🥧',
        'Game Meat': '🦌',
        'Goat Cheese': '🧀',
        'Grilled': '🔥',
        'Ham': '🍖',
        'Hard Cheese': '🧀',
        'Lamb': '🍖',
        'Lasagna': '🍝',
        'Lean Fish': '🐟',
        'Light Stews': '🥘',
        'Maturated Cheese': '🧀',
        'Meat': '🥩',
        'Medium-cured Cheese': '🧀',
        'Mild Cheese': '🧀',
        'Mushrooms': '🍄',
        'Paella': '🥘',
        'Pasta': '🍝',
        'Pizza': '🍕',
        'Pork': '🥓',
        'Poultry': '🍗',
        'Rich Fish': '🐟',
        'Risotto': '🍚',
        'Roast': '🍖',
        'Salad': '🥗',
        'Sashimi': '🍣',
        'Seafood': '🦐',
        'Shellfish': '🦪',
        'Snack': '🥨',
        'Soft Cheese': '🧀',
        'Soufflé': '🥧',
        'Spiced Fruit Cake': '🍰',
        'Spicy Food': '🌶️',
        'Sushi': '🍣',
        'Sweet Dessert': '🍮',
        'Tagliatelle': '🍝',
        'Tomato Dishes': '🍅',
        'Veal': '🥩',
        'Vegetarian': '🥦',
        'Yakissoba': '🍜'
    }
    # Show food options with emoji in dropdown
    food_options_with_emoji = [
        f"{food_emoji_dict.get(food, '')} {food}" for food in unique_foods
    ]
    # Map back from dropdown selection to food name
    selected = st.selectbox("Choose a food:", food_options_with_emoji, key="food_input")
    # Extract food name (remove emoji and space)
    food_input = selected.split(' ', 1)[1] if ' ' in selected else selected

    if st.button("🔎 Recommend Wines"):
        if food_input.strip():
            food_wines = df[df["Harmonize"].apply(
                lambda x: food_input.lower() in [item.lower() for item in ast.literal_eval(x)]
                if isinstance(x, str) and x.startswith("[") else False
            )]
            if not food_wines.empty:
                st.success(f"Found {len(food_wines)} wines for '{food_input}' 🍇")
                top_food_wines = food_wines.head(10)

                for _, row in top_food_wines.iterrows():
                    with st.container():
                        cols = st.columns([1, 4])
                    with cols[0]:
                        st.image("https://purepng.com/public/uploads/large/purepng.com-wine-bottlefood-winebottlealcoholbeverageliquor-2515194557124w46mz.png", width=80)
                    with cols[1]:
                        # Handle grapes
                        grapes = row.get('Grapes') or row.get('Grapes_list', [])
                        if isinstance(grapes, str):
                            try:
                                grapes = ast.literal_eval(grapes)
                            except:
                                grapes = [grapes]
                        grapes_display = ", ".join(grapes) if isinstance(grapes, list) else str(grapes)

                        # Handle harmonize
                        harmonize = row.get("Harmonize", [])
                        if isinstance(harmonize, str):
                            try:
                                harmonize = ast.literal_eval(harmonize)
                            except:
                                pass
                        harmonize_display = ", ".join(harmonize) if isinstance(harmonize, list) else str(harmonize)

                        st.markdown(f"""
                            ### {row['WineName']}
                            - **Grapes**: {grapes_display}
                            - **Body**: {row['Body']}
                            - **ABV**: {row['ABV']}%
                            - **Region**: {row['RegionName']}
                            - **Country**: {row['Country']}
                            - **Food Pairing**: {harmonize_display}
                        """)
                        st.markdown("---")

            else:
                st.warning(f"No wine recommendations found for '{food_input}'.")


# === Main Wine Filters Page ===
if st.session_state.wine_page:
    # === API-Driven Recommendation ===

    st.subheader("🔎 Enter your wine preferences")

    # === Image Upload Box ===
    st.markdown(
        "<h3 style='font-size:1.3rem;'>📸 Upload a Wine Picture</h3>",
        unsafe_allow_html=True
    )
    uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Your uploaded image", use_column_width=True)


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


        country_options = sorted(df["Country"].dropna().unique().tolist())
        country_input = st.selectbox("Country", country_options)
    with col2:

        wine_type_input = st.selectbox("Type", df["Type"].dropna().unique().tolist())

        if country_input:
            region_options = sorted(country_to_regions[country_input])
            region_input = st.selectbox("Region", region_options)
        else:
            region_input = st.selectbox("Region", df["RegionName"].dropna().unique().tolist())

    #     country_input = st.selectbox("Country", df["Country"].dropna().unique() if "Country" in df.columns else [])
    # with col2:
    #     wine_type_input = st.selectbox("Type", df["Type"].dropna().unique() if "Type" in df.columns else [])
    #     region_input = st.selectbox("Region", df["RegionName"].dropna().unique() if "RegionName" in df.columns else [])


    col3, col4 = st.columns(2)
    with col3:
        body_input = st.selectbox("Body", ["Very light-bodied", "Light-bodied", "Medium-bodied", "Full-bodied", "Very full-bodied"])
    with col4:
        abv_input = st.slider("ABV (%)", 5.0, 20.0, 13.5, step=0.1)

    num_recommendations = st.number_input("How many wine recommendations?", 1, 50, 5, step=1)

    if st.button("🔎 Get Recommendations"):
        with st.spinner("Fetching from the CvalVino API..."):
            payload = {
                "wine_type": wine_type_input,
                "grape_varieties": [grape_input] if grape_input else [],
                "body": body_input,
                "abv": abv_input,
                "acidity": "Low",
                "country": country_input,
                "region_name": region_input,
                "n_recommendations": num_recommendations
            }

            try:
                response = requests.post(
                    "https://cvino-api-224355531443.europe-west1.run.app/recommend-wines",
                    json=payload
                )

                if response.status_code == 200:
                    wines = response.json().get("wines", [])
                    if not wines:
                        st.warning("No recommendations found.")
                    else:
                        st.success(f"Here are your top {len(wines)} wine recommendations 🍷")
                        for wine in wines:
                            with st.container():
                                cols = st.columns([1, 4])
                                with cols[0]:
                                    st.image("https://purepng.com/public/uploads/large/purepng.com-wine-bottlefood-winebottlealcoholbeverageliquor-2515194557124w46mz.png", width=80)
                                with cols[1]:
                                    st.markdown(f"""
                                        ### {wine['WineName']}
                                        - **Grapes**: {", ".join(eval(wine['Grapes_list']))}
                                        - **Body**: {wine['Body']}
                                        - **ABV**: {wine['ABV']}%
                                        - **Region**: {wine['RegionName']}
                                        - **Country**: {wine['Country']}
                                        - **Similarity**: {wine['Similarity']:.2f}
                                    """)
                                    st.markdown("---")
                else:
                    st.error(f"API error: {response.status_code} – {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
