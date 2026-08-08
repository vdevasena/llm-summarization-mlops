import sys
import time
from pathlib import Path

import streamlit as st


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# =========================================================
# IMPORT SUMMARIZER
# =========================================================

from summarizer import summarize
from document_loader import extract_text


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Document Summarizer",
    page_icon="📝",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📝 AI Document Summarizer")

st.write(
    "Upload a document or paste text below "
    "to generate an AI-powered summary."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Configuration")

prompt_version = st.sidebar.selectbox(
    "Prompt Version",
    [
        "v1",
        "v2"
    ],
    index=1
)


# =========================================================
# INPUT METHOD
# =========================================================

input_method = st.radio(
    "Choose input method:",
    [
        "Paste text",
        "Upload TXT file"
    ],
    horizontal=True
)


article = ""


# =========================================================
# PASTE TEXT
# =========================================================

if input_method == "Paste text":

    article = st.text_area(
        "Paste your article/document here:",
        height=350,
        placeholder=(
            "Paste the article you want to summarize..."
        )
    )


# =========================================================
# UPLOAD FILE
# =========================================================

else:

    uploaded_file = st.file_uploader(
        "Upload your document",
        type=[
            "txt",
            "md",
            "pdf",
            "docx"
        ]
    )


    if uploaded_file is not None:

        try:

            file_bytes = uploaded_file.read()

            article = extract_text(
                uploaded_file.name,
                file_bytes
            )

            st.text_area(
                "Document preview:",
                article,
                height=300
            )

        except Exception as e:

            st.error(
                f"Could not read the document: {e}"
            )

# =========================================================
# SUMMARIZE
# =========================================================

if st.button(
    "Generate Summary",
    type="primary"
):

    if not article.strip():

        st.warning(
            "Please provide a document first."
        )

    else:

        with st.spinner(
            "Generating summary..."
        ):

            start_time = time.time()

            try:

                summary = summarize(
                    article,
                    prompt_version
                )

                latency = (
                    time.time()
                    - start_time
                )

                st.success(
                    "Summary generated successfully!"
                )

                st.subheader(
                    "Generated Summary"
                )

                st.write(summary)

                st.divider()

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Prompt Version",
                        prompt_version
                    )

                with col2:

                    st.metric(
                        "Latency",
                        f"{latency:.2f} sec"
                    )

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )