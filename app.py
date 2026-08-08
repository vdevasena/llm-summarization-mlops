import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


# =========================================================
# IMPORTS
# =========================================================

from summarizer import summarize

from document_loader import extract_text


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Document Summarizer",
    page_icon="📝",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "📝 AI Document Summarization"
)

st.caption(
    "LLM-powered document summarization with "
    "prompt experimentation and evaluation"
)


# =========================================================
# TABS
# =========================================================

summarize_tab, evaluate_tab = st.tabs(
    [
        "📝 Summarize",
        "📊 Evaluate Prompts"
    ]
)


# =========================================================
# TAB 1 — LIVE SUMMARIZATION
# =========================================================
# =========================================================
# TAB 1 — LIVE SUMMARIZATION
# =========================================================

with summarize_tab:

    st.header(
        "Summarize a Document"
    )

    st.write(
        "Upload a document or paste text and generate "
        "a summary using your selected prompt."
    )

    # -----------------------------------------------------
    # Prompt selection
    # -----------------------------------------------------

    prompt_version = st.selectbox(
        "Choose Prompt Version",
        [
            "v1",
            "v2"
        ]
    )

    # -----------------------------------------------------
    # Input method
    # -----------------------------------------------------

    input_method = st.radio(
        "How would you like to provide the document?",
        [
            "Upload document",
            "Paste text"
        ],
        horizontal=True
    )

    # This will contain the final text
    article = None

    # =====================================================
    # UPLOAD DOCUMENT
    # =====================================================

    if input_method == "Upload document":

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

                st.subheader(
                    "Document"
                )

                st.write(
                    f"**File:** {uploaded_file.name}"
                )

                st.write(
                    f"**Characters:** {len(article):,}"
                )

                with st.expander(
                    "Preview document"
                ):

                    st.text_area(
                        "Extracted Text",
                        article,
                        height=250
                    )

            except Exception as e:

                st.error(
                    f"Unable to process document: {e}"
                )


    # =====================================================
    # PASTE TEXT
    # =====================================================

    else:

        article = st.text_area(
            "Paste your text here",
            height=300,
            placeholder=(
                "Paste the article or document text "
                "you want to summarize..."
            )
        )

        if article:

            st.write(
                f"**Characters:** {len(article):,}"
            )


    # =====================================================
    # GENERATE SUMMARY
    # =====================================================

    # IMPORTANT:
    # This is OUTSIDE the upload/paste blocks.
    # Therefore it works for BOTH input methods.

    if article and article.strip():

        st.divider()

        if st.button(
            "🚀 Generate Summary",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Generating summary..."
            ):

                start_time = time.time()

                result = summarize(
                    article,
                    prompt_version
                )

                latency = (
                    time.time()
                    - start_time
                )

            # -------------------------------------------------
            # Extract results
            # -------------------------------------------------

            summary = result["summary"]

            input_tokens = result["input_tokens"]

            output_tokens = result["output_tokens"]

            total_tokens = result["total_tokens"]

            estimated_cost = result["estimated_cost"]


            # -------------------------------------------------
            # Generated summary
            # -------------------------------------------------

            st.subheader(
                "Generated Summary"
            )

            st.write(
                summary
            )


            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------

            st.subheader(
                "Inference Metrics"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Prompt",
                    prompt_version
                )

            with col2:

                st.metric(
                    "Latency",
                    f"{latency:.2f}s"
                )

            with col3:

                st.metric(
                    "Total Tokens",
                    f"{total_tokens:,}"
                )

            with col4:

                st.metric(
                    "Estimated Cost",
                    f"${estimated_cost:.6f}"
                )


            # -------------------------------------------------
            # Detailed token usage
            # -------------------------------------------------

            st.subheader(
                "Token Usage"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Input Tokens",
                    f"{input_tokens:,}"
                )

            with col2:

                st.metric(
                    "Output Tokens",
                    f"{output_tokens:,}"
                )

            with col3:

                st.metric(
                    "Total Tokens",
                    f"{total_tokens:,}"
                )


    else:

        st.info(
            "Upload a document or paste text "
            "to generate a summary."
        )
        
# =========================================================
# TAB 2 — PROMPT EVALUATION
# =========================================================

with evaluate_tab:

    st.header(
        "📊 Prompt Evaluation"
    )

    st.write(
        "Compare prompt versions using a controlled "
        "evaluation dataset."
    )


    # -----------------------------------------------------
    # Paths
    # -----------------------------------------------------

    RESULTS_DIR = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
    )


    v1_path = (
        RESULTS_DIR
        / "results_v1.csv"
    )

    v2_path = (
        RESULTS_DIR
        / "results_v2.csv"
    )


    # -----------------------------------------------------
    # Check files
    # -----------------------------------------------------

    if (
        not v1_path.exists()
        or not v2_path.exists()
    ):

        st.warning(
            "Evaluation results are not available yet."
        )

        st.write(
            "Run the prompt experiments first:"
        )

        st.code(
            "python src/experiments/run_prompt_experiment.py"
        )


    else:

        # -------------------------------------------------
        # Load results
        # -------------------------------------------------

        v1_df = pd.read_csv(
            v1_path
        )

        v2_df = pd.read_csv(
            v2_path
        )


        # -------------------------------------------------
        # Combine
        # -------------------------------------------------

        comparison_df = pd.concat(
            [
                v1_df,
                v2_df
            ],
            ignore_index=True
        )


        # -------------------------------------------------
        # Average metrics
        # -------------------------------------------------

        summary = (
            comparison_df
            .groupby("prompt_version")
            .agg(
                {
                    "rouge1": "mean",
                    "rouge2": "mean",
                    "rougeL": "mean",
                    "input_tokens": "mean",
                    "output_tokens": "mean",
                    "total_tokens": "mean",
                    "estimated_cost": "mean",
                    "latency_seconds": "mean"
                }
            )
            .reset_index()
        )


        # -------------------------------------------------
        # Results table
        # -------------------------------------------------

        st.subheader(
            "Experiment Results"
        )

        st.dataframe(
            summary,
            use_container_width=True
        )


        # =================================================
        # QUALITY
        # =================================================

        st.subheader(
            "Quality Comparison"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        # -------------------------------------------------
        # ROUGE 1
        # -------------------------------------------------

        rouge1_v1 = summary.loc[
            summary["prompt_version"] == "v1",
            "rouge1"
        ].iloc[0]


        rouge1_v2 = summary.loc[
            summary["prompt_version"] == "v2",
            "rouge1"
        ].iloc[0]


        with col1:

            st.metric(
                "ROUGE-1",
                f"{rouge1_v2:.4f}",
                delta=f"{rouge1_v2 - rouge1_v1:.4f}"
            )


        # -------------------------------------------------
        # ROUGE 2
        # -------------------------------------------------

        rouge2_v1 = summary.loc[
            summary["prompt_version"] == "v1",
            "rouge2"
        ].iloc[0]


        rouge2_v2 = summary.loc[
            summary["prompt_version"] == "v2",
            "rouge2"
        ].iloc[0]


        with col2:

            st.metric(
                "ROUGE-2",
                f"{rouge2_v2:.4f}",
                delta=f"{rouge2_v2 - rouge2_v1:.4f}"
            )


        # -------------------------------------------------
        # ROUGE L
        # -------------------------------------------------

        rougeL_v1 = summary.loc[
            summary["prompt_version"] == "v1",
            "rougeL"
        ].iloc[0]


        rougeL_v2 = summary.loc[
            summary["prompt_version"] == "v2",
            "rougeL"
        ].iloc[0]


        with col3:

            st.metric(
                "ROUGE-L",
                f"{rougeL_v2:.4f}",
                delta=f"{rougeL_v2 - rougeL_v1:.4f}"
            )


        # =================================================
        # COST / PERFORMANCE
        # =================================================

        st.subheader(
            "Cost & Performance"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        # -------------------------------------------------
        # COST
        # -------------------------------------------------

        cost_v1 = summary.loc[
            summary["prompt_version"] == "v1",
            "estimated_cost"
        ].iloc[0]


        cost_v2 = summary.loc[
            summary["prompt_version"] == "v2",
            "estimated_cost"
        ].iloc[0]


        with col1:

            st.metric(
                "Average Cost",
                f"${cost_v2:.6f}",
                delta=f"${cost_v2 - cost_v1:.6f}"
            )


        # -------------------------------------------------
        # TOKENS
        # -------------------------------------------------

        tokens_v1 = summary.loc[
            summary["prompt_version"] == "v1",
            "total_tokens"
        ].iloc[0]


        tokens_v2 = summary.loc[
            summary["prompt_version"] == "v2",
            "total_tokens"
        ].iloc[0]


        with col2:

            st.metric(
                "Average Tokens",
                f"{tokens_v2:,.0f}",
                delta=f"{tokens_v2 - tokens_v1:,.0f}"
            )


        # -------------------------------------------------
        # LATENCY
        # -------------------------------------------------

        latency_v1 = summary.loc[
            summary["prompt_version"] == "v1",
            "latency_seconds"
        ].iloc[0]


        latency_v2 = summary.loc[
            summary["prompt_version"] == "v2",
            "latency_seconds"
        ].iloc[0]


        with col3:

            st.metric(
                "Average Latency",
                f"{latency_v2:.2f}s",
                delta=f"{latency_v2 - latency_v1:.2f}s"
            )


        # =================================================
        # CHARTS
        # =================================================

        st.subheader(
            "Quality Metrics"
        )


        chart_df = (
            summary
            .set_index("prompt_version")
        )


        st.bar_chart(
            chart_df[
                [
                    "rouge1",
                    "rouge2",
                    "rougeL"
                ]
            ]
        )


        st.subheader(
            "Cost & Latency"
        )


        st.bar_chart(
            chart_df[
                [
                    "estimated_cost",
                    "latency_seconds"
                ]
            ]
        )


        # =================================================
        # BEST PROMPT
        # =================================================

        best_prompt = (
            summary
            .sort_values(
                "rougeL",
                ascending=False
            )
            .iloc[0]
        )


        st.success(
            f"🏆 Best prompt based on ROUGE-L: "
            f"{best_prompt['prompt_version']}"
        )