import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import asyncio
from streamlit_option_menu import option_menu

# Multilingual_Summarization.py

# Define a function to load the tokenizer and model asynchronously
async def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum", legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
    return tokenizer, model

# Function to cache the loaded model and tokenizer
async def get_cached_model_and_tokenizer():
    if "cached_model" not in st.session_state:
        st.session_state.cached_model = await load_model_and_tokenizer()
    return st.session_state.cached_model

# Load the model and tokenizer asynchronously
cached_model = asyncio.run(get_cached_model_and_tokenizer())

# Use the cached model and tokenizer for inference
tokenizer, model = cached_model

def summarize(text, max_length=150):
  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
  generated_ids = model.generate(input_ids=input_ids, num_beams=15, num_return_sequences=1, no_repeat_ngram_size=1, remove_invalid_values=True,max_length=max_length,)
  preds = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
  return preds[0]

# on session start
st.set_page_config("NLP Summarization Demo", page_icon=":book:", layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="Summarization",
        options=["Multilingual Summarization","Evaluation"],
        icons=["lightbulb-fill","bar-chart-line-fill"],
        menu_icon="file-earmark-text-fill",
        default_index=0
    )

if selected == "Multilingual Summarization":
    # layout ----------------
    # title and desc
    st.title("Myanmar News Summarization with Multi-language Model")
    st.markdown(
        "This app allows you to test cutting edge news summarization models on \
        text that is in-domain like news from BBC Burmese; and out-of-domain \
        (OOD) such as the Korean Journal Of Medicine and ArXiv."
    )
    # input form area
    st.header("Input Area")
    input_form = st.empty()

    # output area
    col1, col2 = st.columns(2)
    with col1:
        st.header("Full Input")
        full_input = st.empty()

    with col2:
        st.header("Summarization")
        full_output = st.empty()

        # interactivity -----------------
        # set the input form depending on the sidebar input mode
        with input_form.form("Input"):
            text_area = st.text_area("Your Text")
            submitted = st.form_submit_button("Summarize!")
    # run the model on submit
    if submitted:
        text_of_input = text_area
        full_input.markdown(text_of_input)
        # pass input to model and print output
        with st.spinner("Summarizing..."):
            text_of_output = summarize(text_of_input)
            full_output.markdown(text_of_output)
elif selected == "Evaluation":
    st.title(f"You have selected {selected}")


