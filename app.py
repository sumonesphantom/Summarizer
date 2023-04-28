import streamlit as st
from transformers import pipeline
import tokenizers

st.set_page_config(page_title="Summarizer", page_icon=":pencil:", layout="wide")

# Define the hash function for tokenizers.Tokenizer objects
def hash_tokenizer(tokenizer):
    return id(tokenizer)

# Set up the summarization pipeline for English
@st.cache(allow_output_mutation=True,hash_funcs={tokenizers.Tokenizer: hash_tokenizer},suppress_st_warning=True)
def get_summarizer_en():
    return pipeline(
        "summarization",
        model="t5-base",
        tokenizer="t5-base",
        device=0 if st.sidebar.checkbox("Use GPU", key="en_gpu_checkbox", value=True) else -1,
    )

# Set up the summarization pipeline for Hindi
@st.cache(allow_output_mutation=True,hash_funcs={tokenizers.Tokenizer: hash_tokenizer},suppress_st_warning=True)
def get_summarizer_hi():
    return pipeline(
        "summarization",
        model="Helsinki-NLP/opus-mt-en-hi",
        tokenizer="Helsinki-NLP/opus-mt-en-hi",
        device=0 if st.sidebar.checkbox("Use GPU", key="hi_gpu_checkbox", value=True) else -1,
    )

# Define the function for summarizing text
@st.cache(allow_output_mutation=True,hash_funcs={tokenizers.Tokenizer: hash_tokenizer},suppress_st_warning=True)
def summarize(text, language, max_length):
    if language == "English":
        summarizer = get_summarizer_en()
    elif language == "Hindi":
        summarizer = get_summarizer_hi()
    else:
        return "Unsupported language"

    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]["summary_text"]
    return summary

# Define the Streamlit app
def app():
    st.title("Text Summarizer")

    # Sidebar
    language = st.sidebar.selectbox("Language", ["English", "Hindi"])
    use_gpu = st.sidebar.checkbox("Use GPU", value=False)
    max_length = st.sidebar.slider("Max Length", min_value=50, max_value=1000, value=500)

    # Main content
    st.subheader("Enter text to summarize")
    text = st.text_area("Text", height=200)
    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            summary = summarize(text, language, max_length)
        st.subheader("Summary")
        st.write(summary)

    st.sidebar.warning("Note: Using the GPU option requires a CUDA-enabled GPU and compatible drivers.")
    st.sidebar.info("Someonesphantom")

if __name__ == "__main__":
    app()
