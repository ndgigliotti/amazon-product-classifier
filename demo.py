import copy
import json
from functools import partial, lru_cache
import os
from typing import Collection, List
from seaborn import despine
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import wordcloud as wc
import skimage.io
from PIL import Image
import amzsear
from sklearn.preprocessing import minmax_scale
from wordcloud.wordcloud import colormap_color_func

icon = "https://icons-for-free.com/download-icon-Box-1320568095448898951_512.png"

st.set_page_config(
    page_title="Classify a New Amazon Product",
    page_icon=icon,
    layout="centered",
    initial_sidebar_state="auto",
)


def get_coefs(
    model,
    classifier="cls",
    vectorizer="vec",
):
    """Returns labeled model coefficients as a DataFrame."""
    columns = model[vectorizer].get_feature_names()
    coef = pd.DataFrame(
        model[classifier].coef_,
        index=model[classifier].classes_,
        columns=columns,
    )
    return coef.T


def build_search_url(terms: Collection[str]):
    terms = list(terms)
    for i, term in enumerate(terms):
        if "_" in term:
            term = term.replace("_", " ")
            terms[i] = f'"{term}"'
    return amzsear.core.build_url(query="+".join(terms))


def get_keywords(model, text, cat, classifier="cls", vectorizer="vec", drop_neg=False):
    coefs = get_coefs(model, classifier=classifier, vectorizer=vectorizer)
    doc_kws = model[vectorizer].get_keywords(text)
    kw_coefs = coefs.loc[doc_kws.index, cat]
    return doc_kws * kw_coefs


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

    keywords = keywords.to_frame().apply(minmax_scale).squeeze()
    cloud = cloud.generate_from_frequencies(keywords)

    return cloud.to_image()


model = joblib.load("models/final_deploy.joblib")
df = pd.read_parquet("data/flipkart.parquet")
st.markdown(
    f"<div align='center'><img src='{icon}', width=200></img></div>",
    unsafe_allow_html=True,
)
st.markdown("\n" * 2)
st.title("Classify a New Amazon Product")
with open("demo_resources/intro.md") as f:
    st.markdown(f.read())


def random_product():
    options = df.index[df.main_cat == st.session_state.fk_category]
    prev_idx = st.session_state.fk_product_idx
    while st.session_state.fk_product_idx == prev_idx:
        st.session_state.fk_product_idx = np.random.choice(options)


fk_cat_options = np.sort(df["main_cat"].unique())
fk_category = st.select_slider(
    "Select Flipkart category",
    key="fk_category",
    options=fk_cat_options,
    on_change=random_product,
)

if "fk_product_idx" not in st.session_state:
    st.session_state.fk_product_idx = 0


product_idx = st.session_state.fk_product_idx


st.button(
    f"Random Product",
    on_click=random_product,
)
product = df.loc[product_idx]


st.markdown(
    "Feel free to modify or remove parts of the product data, or enter your own."
)

with st.form(key="product_form"):
    title_input = st.text_input(
        label="Enter product title",
        key="title",
        value=product.title,
    )
    brand_input = st.text_input(
        label="Enter product brand",
        key="brand",
        value=product.brand,
    )
    cat_input = st.text_input(
        label="Enter category information",
        key="category",
        value=product.category,
    )
    desc_input = st.text_area(
        label="Enter product description",
        key="description",
        value=product.description,
        height=150,
    )
    classify_button = st.form_submit_button(label="Amazon Classify")


if classify_button:
    combined_text = f"{title_input} {desc_input} {cat_input} {brand_input}"
    pred = model.predict([combined_text])[0]
    keywords = get_keywords(model, combined_text, pred)
    st.header(
        f"Category: {pred.title()}",
    )
    st.markdown("\n")

    img = plot_keywords(keywords)
    st.image(img)
    st.caption(
        "The size of each keyword represents its predictive significance. "
        "Weights are the Hadamard product of the document's TF*IDF vector and the classifier's coefficients."
    )

    top3_kws = keywords.nlargest(3).index.to_list()
    search_kws = build_search_url(top3_kws)
    st.markdown(
        "<div align='center'>"
        f"<a href='{search_kws}', target='_blank'>"
        "Search Keywords on Amazon"
        "</a></div>",
        unsafe_allow_html=True,
    )