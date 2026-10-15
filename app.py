
import re
import json
import gzip
import pickle
from pathlib import Path
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Autocomplete & Autocorrect NLP Dashboard",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


# ============================================================
# 2. DASHBOARD HEADER
# ============================================================

st.title("📝 Autocomplete & Autocorrect NLP Dashboard")

st.caption(
    "Oasis Infobyte | Data Analytics Internship | "
    "Level 2 - Task 5 | WikiText-2"
)

st.markdown(
    "Explore **Bigram and Trigram autocomplete**, "
    "**edit-distance autocorrect**, and NLP model evaluation."
)

st.divider()


# ============================================================
# 3. FILE CHECKING
# ============================================================

REQUIRED_FILES = [
    "model_artifacts.pkl.gz",
    "summary_metrics.json",
    "top20_words.csv"
]

missing_files = [
    filename
    for filename in REQUIRED_FILES
    if not (BASE_DIR / filename).is_file()
]

if missing_files:

    st.error("Some required project files are missing.")

    st.write("### Missing files")

    for filename in missing_files:
        st.write(f"❌ {filename}")

    st.info(
        "Copy the missing files from your Google Colab "
        "task5_outputs.zip into the same folder as app.py."
    )

    with st.expander("Expected project folder structure"):

        st.code(
            """DataAnalytics-L2-AutocompleteAutocorrect/
    app.py
    requirements.txt
    model_artifacts.pkl.gz
    summary_metrics.json
    top20_words.csv
    autocorrect_confusion_matrix.png
    autocorrect_comparison.png""",
            language="text"
        )

    st.stop()


# ============================================================
# 4. LOAD MODEL AND RESULTS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path, modified_time):

    # Load only your own trusted Colab-generated pickle file.
    with gzip.open(model_path, "rb") as file:
        return pickle.load(file)


@st.cache_data(show_spinner=False)
def load_results(metrics_path, words_path):

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    top_words = pd.read_csv(words_path)

    return metrics, top_words


try:

    with st.spinner("Loading trained NLP model..."):

        model_path = BASE_DIR / "model_artifacts.pkl.gz"

        model = load_model(
            str(model_path),
            model_path.stat().st_mtime_ns
        )

        metrics, top_words = load_results(
            str(BASE_DIR / "summary_metrics.json"),
            str(BASE_DIR / "top20_words.csv")
        )

except Exception as error:

    st.error("The NLP model or evaluation files could not be loaded.")

    st.exception(error)

    st.info(
        "Check that the exported model and metrics files "
        "were generated successfully in Google Colab."
    )

    st.stop()


# ============================================================
# 5. VALIDATE MODEL
# ============================================================

required_model_keys = [
    "bigram",
    "trigram",
    "unigrams",
    "fallback",
    "selected_model"
]

missing_keys = [
    key
    for key in required_model_keys
    if key not in model
]

if missing_keys:

    st.error(
        "The model file is incomplete. Missing keys: "
        + ", ".join(missing_keys)
    )

    st.stop()


if not {"Word", "Frequency"}.issubset(top_words.columns):

    st.error(
        "top20_words.csv must contain Word and Frequency columns."
    )

    st.stop()


# ============================================================
# 6. TEXT TOKENIZATION
# ============================================================

def tokenize(text):

    return TOKEN_PATTERN.findall(str(text).lower())


# ============================================================
# 7. AUTOCOMPLETE
# ============================================================

def autocomplete(text, method="trigram", top_k=3):

    words = tokenize(text)

    if not words:
        return []

    predictions = []

    if method == "trigram" and len(words) >= 2:

        context = tuple(words[-2:])

        predictions = model["trigram"].get(context, [])

        if not predictions:

            predictions = model["bigram"].get(
                words[-1], []
            )

    else:

        predictions = model["bigram"].get(
            words[-1], []
        )

    predictions = list(predictions[:top_k])

    for word in model["fallback"]:

        if len(predictions) >= top_k:
            break

        if word not in predictions:
            predictions.append(word)

    return predictions[:top_k]


# ============================================================
# 8. EDIT DISTANCE
# ============================================================

def levenshtein_distance(a, b):

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):

        current = [i]

        for j, char_b in enumerate(b, start=1):

            deletion = previous[j] + 1
            insertion = current[j - 1] + 1

            substitution = (
                previous[j - 1]
                + (char_a != char_b)
            )

            current.append(
                min(
                    deletion,
                    insertion,
                    substitution
                )
            )

        previous = current

    return previous[-1]


# ============================================================
# 9. GENERATE ONE-EDIT CANDIDATES
# ============================================================

def generate_edits(word):

    splits = [
        (word[:i], word[i:])
        for i in range(len(word) + 1)
    ]

    deletes = {
        left + right[1:]
        for left, right in splits
        if right
    }

    transposes = {
        left + right[1] + right[0] + right[2:]
        for left, right in splits
        if len(right) > 1
    }

    replaces = {
        left + char + right[1:]
        for left, right in splits
        if right
        for char in ALPHABET
    }

    inserts = {
        left + char + right
        for left, right in splits
        for char in ALPHABET
    }

    return (
        deletes
        | transposes
        | replaces
        | inserts
    )


# ============================================================
# 10. AUTOCORRECT
# ============================================================

vocabulary = {
    word: count
    for word, count in model["unigrams"].items()
    if count >= 2
    and word.isalpha()
    and len(word) >= 2
}


def autocorrect(word):

    tokens = tokenize(word)

    if not tokens:
        return ""

    word = tokens[0]

    if word in vocabulary:
        return word

    candidates = (
        generate_edits(word)
        & vocabulary.keys()
    )

    if not candidates:
        return word

    return min(
        candidates,
        key=lambda candidate: (
            levenshtein_distance(word, candidate),
            -vocabulary[candidate],
            candidate
        )
    )


# ============================================================
# 11. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    selected_default = model.get(
        "selected_model",
        "trigram"
    )

    method = st.selectbox(
        "Autocomplete Model",
        ["trigram", "bigram"],
        index=0 if selected_default == "trigram" else 1
    )

    top_k = st.slider(
        "Number of Predictions",
        min_value=1,
        max_value=5,
        value=3
    )

    st.divider()

    st.subheader("Project Information")

    st.write("**Dataset:** WikiText-2")
    st.write("**Task:** Autocomplete & Autocorrect")
    st.write("**Model:** Frequency-based NLP")

    st.success("Model loaded successfully!")


# ============================================================
# 12. MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "✨ Autocomplete",
    "✍️ Autocorrect",
    "📊 Model Evaluation",
    "📚 Corpus Insights"
])


# ============================================================
# TAB 1: AUTOCOMPLETE
# ============================================================

with tab1:

    st.subheader("✨ Next Word Prediction")

    st.write(
        "Enter a phrase to predict the next word "
        "using the selected N-gram model."
    )

    prefix = st.text_input(
        "Enter text prefix",
        value="in the",
        key="autocomplete_input"
    )

    if prefix.strip():

        suggestions = autocomplete(
            prefix,
            method,
            top_k
        )

        st.markdown("### Top Predictions")

        if suggestions:

            columns = st.columns(len(suggestions))

            for index, suggestion in enumerate(suggestions):

                with columns[index]:

                    st.metric(
                        f"Prediction {index + 1}",
                        suggestion
                    )

            st.markdown("### Completed Text Examples")

            for suggestion in suggestions:

                st.write(
                    f"**{prefix.strip()} {suggestion}**"
                )

        else:

            st.warning("No predictions available.")

    else:

        st.info("Please enter a text prefix.")

    st.divider()

    st.caption(
        "Example inputs: in the | one of | "
        "it was | during the | the first"
    )


# ============================================================
# TAB 2: AUTOCORRECT
# ============================================================

with tab2:

    st.subheader("✍️ Spelling Correction")

    st.write(
        "Enter a misspelled word to generate "
        "an edit-distance-based correction."
    )

    input_word = st.text_input(
        "Enter a word",
        value="computr",
        key="autocorrect_input"
    )

    if input_word.strip():

        corrected = autocorrect(input_word)

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Original Input",
                input_word
            )

        with col2:

            st.metric(
                "Suggested Correction",
                corrected if corrected else "N/A"
            )

        normalized = tokenize(input_word)

        if normalized:

            if corrected == normalized[0]:

                st.info(
                    "The word is already recognized, "
                    "or no suitable one-edit correction "
                    "was found."
                )

            else:

                st.success(
                    f"Suggested correction: {corrected}"
                )

    else:

        st.info("Enter a word to check its spelling.")

    st.divider()

    st.caption(
        "The current model mainly handles "
        "single-edit spelling mistakes."
    )


# ============================================================
# TAB 3: MODEL EVALUATION
# ============================================================

with tab3:

    st.subheader("📊 NLP Model Evaluation")

    # ------------------------------
    # Autocomplete metrics
    # ------------------------------

    st.markdown("### Bigram vs Trigram")

    auto_results = pd.DataFrame(
        metrics.get("test_autocomplete", [])
    )

    required_auto_columns = {
        "Model",
        "Precision_at_3",
        "Recall_at_3"
    }

    if (
        not auto_results.empty
        and required_auto_columns.issubset(
            auto_results.columns
        )
    ):

        st.dataframe(
            auto_results,
            use_container_width=True,
            hide_index=True
        )

        comparison = auto_results.melt(
            id_vars="Model",
            value_vars=[
                "Precision_at_3",
                "Recall_at_3"
            ],
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            comparison,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Autocomplete Model Comparison",
            labels={"Score": "Metric Score"}
        )

        fig.update_layout(
            yaxis_range=[0, 1]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Precision@3 measures relevant predictions "
            "among three suggestions. Recall@3 measures "
            "whether the actual next word appears in "
            "the top three suggestions."
        )

    else:

        st.warning(
            "Autocomplete evaluation metrics are unavailable."
        )

    st.divider()

    # ------------------------------
    # Autocorrect metrics
    # ------------------------------

    st.markdown("### Autocorrect Evaluation")

    spell_results = pd.DataFrame(
        metrics.get("autocorrect", [])
    )

    if not spell_results.empty:

        st.dataframe(
            spell_results,
            use_container_width=True,
            hide_index=True
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        typo_accuracy = metrics.get(
            "typo_only_accuracy"
        )

        if typo_accuracy is not None:

            st.metric(
                "Exact Typo Correction Accuracy",
                f"{typo_accuracy:.2%}"
            )

    with col2:

        detection_precision = metrics.get(
            "correction_detection_precision"
        )

        if detection_precision is not None:

            st.metric(
                "Correction Detection Precision",
                f"{detection_precision:.2%}"
            )

    with col3:

        detection_recall = metrics.get(
            "correction_detection_recall"
        )

        if detection_recall is not None:

            st.metric(
                "Correction Detection Recall",
                f"{detection_recall:.2%}"
            )

    st.info(
        "Correction detection precision and recall "
        "measure whether the model attempts a correction "
        "when needed. They do not measure whether the "
        "replacement word is correct. Exact typo "
        "correction accuracy measures that separately."
    )

    # ------------------------------
    # Autocorrect comparison chart
    # ------------------------------

    if (
        not spell_results.empty
        and {
            "Model",
            "Accuracy",
            "Weighted_Precision",
            "Weighted_Recall"
        }.issubset(spell_results.columns)
    ):

        spell_chart = spell_results.melt(
            id_vars="Model",
            value_vars=[
                "Accuracy",
                "Weighted_Precision",
                "Weighted_Recall"
            ],
            var_name="Metric",
            value_name="Score"
        )

        fig_spell = px.bar(
            spell_chart,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Autocorrect vs Identity Baseline"
        )

        fig_spell.update_layout(
            yaxis_range=[0, 1]
        )

        st.plotly_chart(
            fig_spell,
            use_container_width=True
        )

    # ------------------------------
    # Confusion matrix
    # ------------------------------

    st.markdown("### Autocorrect Confusion Matrix")

    matrix_path = (
        BASE_DIR / "autocorrect_confusion_matrix.png"
    )

    if matrix_path.exists():

        st.image(
            str(matrix_path),
            caption=(
                "Correctness Before vs After Autocorrect"
            ),
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion matrix image not found. "
            "Copy autocorrect_confusion_matrix.png "
            "from your Colab outputs folder."
        )


# ============================================================
# TAB 4: CORPUS INSIGHTS
# ============================================================

with tab4:

    st.subheader("📚 WikiText Corpus Analysis")

    st.markdown("### Top 20 Most Frequent Words")

    words_chart = px.bar(
        top_words.sort_values("Frequency"),
        x="Frequency",
        y="Word",
        orientation="h",
        title="Top 20 Most Frequent Words",
        color="Frequency",
        color_continuous_scale="Blues"
    )

    words_chart.update_layout(
        coloraxis_showscale=False,
        height=650
    )

    st.plotly_chart(
        words_chart,
        use_container_width=True
    )

    st.divider()

    st.markdown("### Project Limitations")

    st.markdown(
        """
        - N-gram models capture limited context.
        - The corpus contains encyclopedic rather than
          conversational language.
        - Edit-distance correction may fail on multiple
          spelling mistakes.
        - Real-word errors require contextual correction.
        - Synthetic typo results may not represent
          real-world keyboard usage.
        """
    )

    st.markdown("### Future Improvements")

    st.markdown(
        """
        - Transformer-based autocomplete.
        - Context-aware spelling correction.
        - Larger and more diverse training corpora.
        - Evaluation using real typing mistakes.
        - Improved handling of unknown vocabulary.
        """
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "Oasis Infobyte | Data Analytics Internship | "
    "Developed by: [Shubham Maity] | 2026 | "
    "Level 2 Task 5 — Autocomplete and Autocorrect "
    "Data Analytics"
)
