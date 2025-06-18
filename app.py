import streamlit as st
import pandas as pd
import os
import ast
import requests
from cv_functions.wine_label_ai2 import extract_wine_info_from_image
import ipdb
import tempfile

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

# create grape varieties list
all_grapes = []
for grapes in df["Grapes_list"]:
    # Convert string representation of list to actual list if needed
    if isinstance(grapes, str):
        grapes_list = ast.literal_eval(grapes)
    else:
        grapes_list = grapes
    all_grapes.extend([g.strip() for g in grapes_list])

unique_grapes = sorted(set(all_grapes))


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
    selected_foods = st.multiselect(
        "Food (start typing for suggestions)",
        options=food_options_with_emoji,
        help="Start typing to select one or more foods from our database."
    )

    wine_types = df["Type"].dropna().unique().tolist()
    wine_type_selected = st.selectbox("🍷 Prefer a wine type?", wine_types)

    # Extract food names (remove emoji and space)
    food_inputs = [s.split(' ', 1)[1] if ' ' in s else s for s in selected_foods]
    # For compatibility with the rest of the code, join as comma-separated string
    food_input = ", ".join(food_inputs)

    if st.button("🔎 Recommend Wines"):
        if selected_foods:
            selected_food_names = [food.split(' ', 1)[1] if ' ' in food else food for food in selected_foods]

            def matches_any_food(harmonize_str):
                if isinstance(harmonize_str, str) and harmonize_str.startswith("["):
                    try:
                        harmonized_foods = [item.lower() for item in ast.literal_eval(harmonize_str)]
                        return any(food.lower() in harmonized_foods for food in selected_food_names)
                    except:
                        return False
                return False

            food_wines = df[df["Harmonize"].apply(matches_any_food)]

            if wine_type_selected != "All":
                food_wines = food_wines[food_wines["Type"] == wine_type_selected]

                if not food_wines.empty:
                    st.success(f"Found {len(food_wines)} wines for '{food_input}' 🍇")

                    top_food_wines = food_wines.head(10)

                    for _, row in top_food_wines.iterrows():
                        with st.container():
                            cols = st.columns([1, 4])

                            with cols[0]:
                                st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
                                # Show wine type image
                                if row['Type'] == "Red":
                                    image_path = "images/red_wine.png"
                                elif row['Type'] == "White":
                                    image_path = "images/white_wine.png"
                                elif row['Type'] == "Rosé":
                                    image_path = "images/rose_wine.png"
                                elif row['Type'] == "Sparkling":
                                    image_path = "images/sparkling_wine.png"
                                elif row['Type'] == "Dessert":
                                    image_path = "images/dessert.png"
                                elif row['Type'] == "Dessert/Port":
                                    image_path = "images/port.png"

                                st.image(image_path, width=250)

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
                                    - **Type**: {row['Type']}
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

            else:
                st.warning(f"No wine recommendations found for '{food_input}'.")


# === Main Wine Filters Page ===
if st.session_state.wine_page:
    # === API-Driven Recommendation ===

    st.subheader("🔎 Enter your wine preferences")

    # === Acidity Default ===
    acidity_input = "Medium"

    # === Image Upload Box with Side-by-Side Layout ===
    st.markdown(
        "<h3 style='font-size:1.3rem;'>📸 Upload a Wine Picture</h3>",
        unsafe_allow_html=True
    )
    upload_col, img_col = st.columns([1, 2])
    with upload_col:
        uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    with img_col:
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Your uploaded image", width=150)

    # Always show the button if an image is uploaded
    send_to_api_clicked = st.button("Send to API", disabled=(uploaded_image is None))

    if send_to_api_clicked and uploaded_image is not None:
        # Send image to FastAPI
        files = {'img': (uploaded_image.name, uploaded_image, uploaded_image.type)}
        response = requests.post("https://cvino-api-224355531443.europe-west1.run.app/read_image", files=files)
        #response = requests.post("http://localhost:8502/", files=files)
        if response.status_code == 200:
            st.success("Image successfully uploaded and processed!")

            wine_info = response.json()
            wine_type_input = wine_info.get('wine_type')
            grape_input = wine_info.get('grape_varieties')
            body_input = wine_info.get('body')
            acidity_input = wine_info.get('acidity')
            country_input = wine_info.get('country')
            region_input = wine_info.get('region')
            abv_input = wine_info.get('ABV')

            #st.markdown(win_info.get('message', 'Wine information extracted successfully!'), unsafe_allow_html=True)

        else:
            st.error(f"Failed to upload image: {response.json().get('message')}")


    col1, col2 = st.columns(2)
    with col1:
        # Autocomplete grape input using unique_grapes
        grape_input = st.multiselect(
            "Grape (start typing for suggestions)",
            options=unique_grapes,
            help="Start typing to select one or more grape varieties from our database."
        )
        # Convert multiselect list to comma-separated string (like text_input)
        if grape_input:
            grape_input = ", ".join(grape_input)
        else:
            grape_input = ""


        wine_name_input = None

        country_options = sorted(df["Country"].dropna().unique().tolist())
        country_input = st.selectbox("Country", country_options)
    with col2:

        wine_type_input = st.selectbox("Type", df["Type"].dropna().unique().tolist(), index=1)

        if country_input:
            region_options = sorted(country_to_regions[country_input])
            region_input = st.selectbox("Region", region_options)
        else:
            region_input = st.selectbox("Region", df["RegionName"].dropna().unique().tolist())


    col3, col4 = st.columns(2)
    with col3:
        body_input = st.selectbox(
            "Body",
            ["Very light-bodied", "Light-bodied", "Medium-bodied", "Full-bodied", "Very full-bodied"],
            index=2  # 0-based index, so 2 is "Medium-bodied"
        )
    with col4:
        abv_input = st.slider("ABV (%)", 5.0, 20.0, 13.5, step=0.1)

    num_recommendations = st.number_input("How many wine recommendations?", 1, 50, 5, step=1)

    if st.button("🔎 Get Recommendations"):
        st.components.v1.html("""
        <audio autoplay id="temp-audio">
            <source src="https://raw.githubusercontent.com/Obispodino/cvino/master/interface/wine-music.mp3" type="audio/mp3">
        </audio>
        """, height=0)

        with st.spinner("Fetching from the CvalVino API..."):
            payload = {
                "wine_type": wine_type_input,
                "grape_varieties": [grape_input] if grape_input else [],
                "body": body_input,
                "abv": abv_input,
                "acidity": acidity_input,
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
                        # 2. Stop audio
                         st.components.v1.html("""
                        <script>
                            const audio = document.getElementById("temp-audio");
                            if (audio) {
                                audio.pause();
                                audio.currentTime = 0;
                                audio.remove();
                            }
                        </script>
                        """, height=0)
                    st.success(f"Here are your top {len(wines)} wine recommendations 🍷")

                    for wine in wines:
                        with st.container():
                            cols = st.columns([1, 4])
                            with cols[0]:
                                # Add vertical space above the image to lower its position
                                st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
                                if wine['Type'] == "Red":
                                    image_path = "images/red_wine.png"
                                elif wine['Type'] == "White":
                                    image_path = "images/white_wine.png"
                                elif wine['Type'] == "Rosé":
                                    image_path = "images/rose_wine.png"
                                elif wine['Type'] == "Sparkling":
                                    image_path = "images/sparkling_wine.png"
                                elif wine['Type'] == "Dessert":
                                    image_path = "images/dessert.png"
                                elif wine['Type'] == "Dessert/Port":
                                    image_path = "images/port.png"
                                st.image(image_path, width=250)

                            with cols[1]:
                                st.markdown(f"""
                                    ### {wine['WineName']}
                                    - **Type**: {wine['Type']}
                                    - **Grapes**: {", ".join(ast.literal_eval(wine['Grapes_list'])) if isinstance(wine['Grapes_list'], str) else wine['Grapes_list']}
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
