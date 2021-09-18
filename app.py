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


def extract_coef(
    model,
    classifier="cls",
    vectorizer="vec",
):
    """Returns labeled model coefficients as a DataFrame."""
    columns = np.array(model[vectorizer].get_feature_names())
    coef = pd.DataFrame(
        model[classifier].coef_,
        index=model[classifier].classes_,
        columns=columns,
    )
    return coef.T


def get_keywords(model, text, cat, classifier="cls", vectorizer="vec", drop_neg=False):
    coef = extract_coef(model, classifier=classifier, vectorizer=vectorizer)
    raw_kw = model[vectorizer].get_keywords(text)
    kw_coef = coef.loc[raw_kw.index, cat]
    kw = raw_kw * kw_coef
    if not drop_neg:
        if (kw < 0).any():
            kw += kw.min() + np.finfo(np.float64).min
    else:
        kw = kw.loc[kw > 0]
    return kw


def plot_keywords(keywords, size=(1000, 700), cmap="magma", random_state=350):

    cloud = wc.WordCloud(
        colormap=cmap,
        width=size[0],
        height=size[1],
        random_state=random_state,
        mode="RGBA",
        background_color=None,
        color_func=colormap_color_func(cmap),
        repeat=False,
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
    classify_button = st.form_submit_button(label="Amazon Classify")

# st.form_submit_button returns True upon form submit
if classify_button:
    combined_text = f"{title_input} {desc_input} {cat_input} {brand_input}"
    pred = model.predict([combined_text])[0]
    st.header(f"Category: {pred.title()}")
    st.text("\n" * 2)
    st.subheader("Keyword Importance")
    st.markdown("The size of each keyword represents its predictive significance. ")
    st.text("\n" * 2)
    keywords = get_keywords(model, combined_text, pred)
    img = plot_keywords(keywords)
    st.image(img)
    st.caption(
        "The size weights are obtained (roughly) by multiplying the document's "
        f"TF*IDF vector by the classifier's coefficients."
    )