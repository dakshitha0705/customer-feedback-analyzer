"""Streamlit entry point: `streamlit run app/main.py`"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402  (installed with streamlit)
import streamlit as st  # noqa: E402

from app import core  # noqa: E402

st.set_page_config(page_title="Customer Feedback Analyzer")
st.title("Customer Feedback Analyzer")
st.write("Paste customer comments (one per line). A fine-tuned distilgpt2 sorts each into "
         "**product issue**, **delivery / shipping**, **staff praise** or **pricing / value**.")


@st.cache_resource
def get_model():
    return core.load_model()


loaded = get_model()
if loaded is None:
    st.warning("Model not trained yet. Run `python -m app.core train` (about a minute on CPU), then reload.")
    st.stop()
tokenizer, model = loaded

example = ("The headphones stopped charging after a week.\n"
           "My parcel was left at the wrong house.\n"
           "Jess at the till was lovely and so helpful.\n"
           "Way too pricey for such a small bottle.")
text = st.text_area("Comments", value=example, height=180)

if st.button("Classify", type="primary"):
    comments = core.parse_comments(text)
    if not comments:
        st.info("Enter at least one comment.")
    else:
        rows = core.classify_many(comments, tokenizer, model)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.subheader("Category counts")
        st.bar_chart(pd.Series(core.category_counts(rows), name="comments"))
