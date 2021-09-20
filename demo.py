from typing import Collection, NoReturn, Tuple

import amzsear
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import wordcloud as wc
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from PIL.Image import Image
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.preprocessing import minmax_scale
from sklearn.utils.validation import check_is_fitted
from wordcloud.wordcloud import colormap_color_func

from tools import utils

icon = "https://icons-for-free.com/download-icon-Box-1320568095448898951_512.png"

st.set_page_config(
    page_title="Classify a New Amazon Product",
    page_icon=icon,
    layout="centered",
    initial_sidebar_state="auto",
)


def get_search_url(terms: Collection[str]) -> str:
    """Constructs Amazon search URL.

    Parameters
    ----------
    terms : Collection[str]
        Terms to use in query. Complex terms with '_' will be
        split and placed in quotes.

    Returns
    -------
    str
        Amazon search URL.
    """
    terms = list(terms)
    for i, term in enumerate(terms):
        if "_" in term:
            term = term.replace("_", " ")
            terms[i] = f'"{term}"'
    return amzsear.core.build_url(query="+".join(terms))


def classify(
    text: str,
    vectorizer: _VectorizerMixin,
    classifier: BaseEstimator,
) -> Tuple[str, Series]:
    """Classify text and get scored keywords.

    Keyword scores are the Hadamard product of document's TF*IDF
    vector and the correspoding coefficients for the predicted class.

    Parameters
    ----------
    text : str
        Classify and get keywords for this document.
    vectorizer : _VectorizerMixin, optional
        Fitted vectorizer with `vocabulary_`.
    classifier : str, optional
        Fitted classifier with `coef_`.


    Returns
    -------
    str, Series
        Category and scored keywords.
    """
    check_is_fitted(vectorizer, "vocabulary_")
    check_is_fitted(classifier, "coef_")

    # Compute vector, predict class
    vector = vectorizer.transform([text])
    category = classifier.predict(vector)[0]

    # Label coefficients
    coefs = DataFrame(
        classifier.coef_,
        index=classifier.classes_,
        columns=vectorizer.get_feature_names(),
    )
    # Get labeled non-zero TF*IDF scores
    vocab = utils.swap_index(Series(vectorizer.vocabulary_))
    tfidf_kws = Series(vector.data, index=vocab.loc[vector.indices], name="keywords")

    # Multiply TF*IDF scores by coefficients
    keywords = tfidf_kws * coefs.loc[category, tfidf_kws.index]
    return category, keywords


def plot_keywords(
    keywords: Series,
    size: Tuple[int, int] = (1000, 700),
    cmap: str = "magma",
    random_state: int = 350,
) -> Image:
    """Makes wordcloud from keywords with frequencies.

    Parameters
    ----------
    keywords : Series
        Series of frequencies indexed by keyword. Will be scaled to [0, 1].
    size : Tuple[int, int], optional
        Image (width, height) in pixels, by default (1000, 700).
    cmap : str, optional
        Name of matplotlib colormap to use, by default "magma".
    random_state : int, optional
        Integer seed for reproducibility across calls, by default 350.

    Returns
    -------
    Image
        Wordcloud.
    """
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

    # Scale keywords and generate cloud
    keywords = keywords.to_frame().apply(minmax_scale).squeeze()
    cloud = cloud.generate_from_frequencies(keywords)

    return cloud.to_image()


# Load model if necessary
if "model" in st.session_state:
    model = st.session_state.model
else:
    model = joblib.load("models/final_deploy.joblib")
    st.session_state.model = model

# Load data if necessary
if "wm_data" in st.session_state:
    df = st.session_state.wm_data
else:
    df = pd.read_parquet("data/walmart.parquet")
    st.session_state.wm_data = df

# Image/icon at top of page
st.markdown(
    f"<div align='center'><img src='{icon}', width=200></img></div>",
    unsafe_allow_html=True,
)
st.markdown("\n" * 2)

st.title("Classify a New Amazon Product")

# Load intro markdown if necessary
if "intro" not in st.session_state:
    with open("demo_resources/intro.md") as f:
        st.session_state.intro = f.read()

# Intro markdown
st.markdown(st.session_state.intro)


def random_product() -> NoReturn:
    """Select a new random product within current category."""
    options = df.index[df.main_cat == st.session_state.wm_category]
    if "wm_product_idx" in st.session_state:
        prev_idx = st.session_state.wm_product_idx
        while st.session_state.wm_product_idx == prev_idx:
            st.session_state.wm_product_idx = np.random.choice(options)
    else:
        st.session_state.wm_product_idx = np.random.choice(options)


# Category slider
wm_cat_options = np.sort(df["main_cat"].unique())
wm_category = st.select_slider(
    "Select Walmart category",
    key="wm_category",
    options=wm_cat_options,
    on_change=random_product,
)

# Select first product
if "wm_product_idx" not in st.session_state:
    random_product()


product_idx = st.session_state.wm_product_idx

# Button to select random new product
st.button(
    f"Random Product",
    on_click=random_product,
)
product = df.loc[product_idx]


# Editable product data
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

# Classify text and plot keywords
if classify_button:
    combined_text = f"{title_input} {desc_input} {cat_input} {brand_input}"
    pred, keywords = classify(combined_text, model["vec"], model["cls"])
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

    # Amazon search link
    top3_kws = keywords.nlargest(3).index.to_list()
    search_kws = get_search_url(top3_kws)
    st.markdown(
        "<div align='center'>"
        f"<a href='{search_kws}', target='_blank'>"
        "Search Keywords on Amazon"
        "</a></div>",
        unsafe_allow_html=True,
    )
