import json
from functools import partial
from seaborn import despine
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import wordcloud as wc
from wordcloud.wordcloud import colormap_color_func
from tools import language as lang, plotting

rng = np.random.default_rng()


def plot_keywords(
    keywords, size=(1000, 700), cmap="magma", title_size=16, random_state=350
):
    # keywords.index = keywords.index.str.replace("_", " ", regex=False)

    cloud = wc.WordCloud(
        colormap=cmap,
        width=size[0],
        height=size[1],
        random_state=random_state,
        mode="RGBA",
        background_color=None,
        color_func=colormap_color_func(cmap),
    )

    cloud = cloud.generate_from_frequencies(keywords)

    return cloud.to_image()


model = joblib.load("models/final_deploy.joblib")
fk = pd.read_parquet("data/flipkart.parquet")
fk = fk.loc[fk.n_char >= fk.n_char.median()]

st.title("Classify a New Amazon Product")
st.markdown(
    "Enter your own product data, or select a product from the provided dataset. "
    "The following data comes from [Flipkart](https://www.flipkart.com/), an Indian competetor of Amazon. "
    "It was [scraped](https://www.kaggle.com/PromptCloudHQ/flipkart-products) in 2016."
)

main_cat_opt = fk["main_cat"].dropna().sort_values().unique()
main_cat = st.select_slider(
    "Select Flipkart category", options=main_cat_opt, value=main_cat_opt[0]
)
fk = fk.loc[fk.main_cat == main_cat]
st.dataframe(fk.loc[:, ["title", "category", "description", "brand"]])
prod_idx = st.select_slider(
    "Select Flipkart product", options=fk.index, value=fk.index[0]
)

product = fk.loc[prod_idx]
st.markdown("Feel free to modify or remove parts of the product data.")
# Forms can be declared using the 'with' syntax
with st.form(key="my_form"):
    title_input = st.text_input(label="Enter product title", value=product["title"])
    brand_input = st.text_input(label="Enter product brand", value=product["brand"])
    cat_input = st.text_input(
        label="Enter category information", value=product["category"]
    )
    desc_input = st.text_area(
        label="Enter product description",
        value=product["description"],
        height=150,
    )
    classify_button = st.form_submit_button(label="Classify")

# st.form_submit_button returns True upon form submit
if classify_button:
    combined_text = f"{title_input} {desc_input} {cat_input} {brand_input}"
    pred = model.predict([combined_text])[0]
    st.header(f"Category: {pred.title()}")
    st.text("\n" * 2)
    st.subheader("Top Keywords in Model")
    st.text("\n" * 2)
    keywords = model["vec"].get_keywords(combined_text)
    img = plot_keywords(keywords)
    st.image(img)
