import streamlit as st
from fastai.vision.all import *
from util import predict
import pandas as pd 

MODEL_PATH = 'model_name.pkl'
ITEM = "Example Item"

# ------------ Load your Model ------------ #
learn = load_learner(MODEL_PATH)
labels = learn.dls.vocab


# ------------ Build UI ------------ #
st.set_page_config(layout="wide")
file_upload, output = st.column(2, gap='large')

st.markdown("---")

with file_upload:
    st.subheader("Upload a Watch PNG or JPG")
    st.file_upload(f"Upload a {ITEM} (.png / .jpg)")

with output:
    if 'watch_img' not in st.session_state or st.session_state["watch_img"] is None:
        st.warning(f"Upload a {ITEM} image")
    
    else:
        pred, wmap = predict(
            st.session_state["watch_img"],
            learner,
            labels
        )
        st.header(f"The type of {ITEM} is: {pred}")
        pred_df = pd.DataFrame(wmap.items(), columns=["Prediction", "Confidence"])\
                                .sort_values('Confidence', ascending=False)
        st.bar_chart(pred_df, x=ITEM, y="Confidence")
