
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MutAI | Protein Mutation Analysis",
    page_icon="🧬",
    layout="wide"
)

# ============================================================
# PASTEL UI
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #fff7fb, #f8f5ff, #f3fbff);
}

h1, h2, h3 {
    color: #263859;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    color: #263859;
}

.subtitle {
    font-size: 1.1rem;
    color: #667085;
}

div[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.75);
    border-radius: 18px;
    padding: 15px;
    border: 1px solid #eadff0;
}

.stButton > button {
    border-radius: 14px;
    border: 1px solid #e6cfe0;
    background-color: #fff0f7;
    color: #263859;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "mutation_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "target_scaler.pkl")
INFO_PATH = os.path.join(BASE_DIR, "model_info.pkl")

# ============================================================
# LOAD FILES
# ============================================================

@st.cache_resource
def load_pickle(path):
   return joblib.load(path)

try:
    model = load_pickle(MODEL_PATH)
    feature_columns = load_pickle(FEATURE_PATH)
    target_scaler = load_pickle(SCALER_PATH)
    model_info = load_pickle(INFO_PATH)

    model_loaded = True

except Exception as e:
    model_loaded = False
    load_error = str(e)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧬 MutAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Protein Mutation Analysis Platform'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "An ML-based platform for exploring protein substitutions "
    "and predicting their DMS scores."
)

st.divider()

# ============================================================
# DATASET OVERVIEW
# ============================================================

st.subheader("🌸 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("DMS datasets", "217")

with c2:
    st.metric("Total records", "2.46M")

with c3:
    st.metric("Substitutions", "696K")

with c4:
    st.metric("ML features", "41")

st.divider()

# ============================================================
# MODEL STATUS
# ============================================================

if not model_loaded:

    st.error("⚠️ Model files could not be loaded.")

    st.code(load_error)

    st.stop()

# ============================================================
# MUTATION PREDICTOR
# ============================================================

st.subheader("🎀 Mutation Impact Predictor")

st.write(
    "Enter a single amino-acid substitution to obtain "
    "an ML-based predicted DMS score."
)

amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

c1, c2, c3 = st.columns(3)

with c1:
    original = st.selectbox(
        "Original amino acid",
        amino_acids,
        index=amino_acids.index("K")
    )

with c2:
    position = st.number_input(
        "Mutation position",
        min_value=1,
        value=291,
        step=1
    )

with c3:
    mutant = st.selectbox(
        "Mutant amino acid",
        amino_acids,
        index=amino_acids.index("V")
    )

mutation = f"{original}{int(position)}{mutant}"

if st.button("🔮 Predict Mutation", use_container_width=True):

    if original == mutant:
        st.warning(
            "Please choose a different mutant amino acid."
        )

    else:

        # ----------------------------------------------------
        # CREATE FEATURE VECTOR
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            np.zeros((1, len(feature_columns))),
            columns=feature_columns
        )

        original_col = f"original_{original}"
        mutant_col = f"new_{mutant}"

        if original_col in input_df.columns:
            input_df.loc[0, original_col] = 1

        if mutant_col in input_df.columns:
            input_df.loc[0, mutant_col] = 1

        if "position" in input_df.columns:
            input_df.loc[0, "position"] = position

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        try:

            prediction_normalized = model.predict(input_df)[0]

            prediction_original = (
                prediction_normalized *
                target_scaler.scale_[0]
                +
                target_scaler.mean_[0]
            )

            st.success(f"Prediction completed for **{mutation}** 🎀")

            r1, r2 = st.columns(2)

            with r1:
                st.metric(
                    "Predicted normalized score",
                    f"{prediction_normalized:.4f}"
                )

            with r2:
                st.metric(
                    "Predicted DMS score",
                    f"{prediction_original:.2f}"
                )

            st.info(
                "This prediction estimates the mutation's DMS score "
                "using the trained ML model. It should be interpreted "
                "as a computational prediction rather than an "
                "experimental measurement."
            )

        except Exception as e:

            st.error("Prediction failed.")

            st.code(str(e))

st.divider()

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("📊 Model Performance")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("R²", "0.0578")

with c2:
    st.metric("Training samples", "557,048")

with c3:
    st.metric("Testing samples", "139,263")

st.info(
    "The current model is a baseline predictor using mutation "
    "identity and position as input features. Its relatively low "
    "R² indicates that additional biological features such as "
    "sequence context or structural information could improve "
    "predictive performance."
)

st.divider()

# ============================================================
# FOOTER
# ============================================================

st.caption(
    "🧬 MutAI • Protein Mutation Analysis Platform • Research Prototype"
)
