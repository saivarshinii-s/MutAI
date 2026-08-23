
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import zipfile
import re
import time
from pathlib import Path

# ============================================================
# STRUCTURAL ANALYSIS FILES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALPHAFOLD_ZIP = os.path.join(BASE_DIR, "AlphaFold_MutAI.zip")
DYNAMUT2_PDB = os.path.join(BASE_DIR, "1cei_DynaMut2.pdb")
FOLDX_REPAIR_PDB = os.path.join(BASE_DIR, "1cei_Repair.pdb")
FOLDX_REPAIR_FXOUT = os.path.join(BASE_DIR, "1cei_Repair.fxout")

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

            st.session_state["active_mutation"] = mutation
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
# MUTATION-SPECIFIC STRUCTURAL ANALYSIS
# ============================================================

# Keep the mutation selected by the user across Streamlit reruns.
active_mutation = st.session_state.get("active_mutation")

st.divider()
st.header("🧬 Structural Analysis")

if not active_mutation:
    st.info(
        "Enter a mutation above and click **🔮 Predict Mutation**. "
        "The selected mutation will then be used for the structural-analysis inputs below."
    )
else:
    st.success(f"🎯 Active mutation: **{active_mutation}**")

    # --------------------------------------------------------
    # FOLDX
    # --------------------------------------------------------
    st.subheader(f"🧪 FoldX — {active_mutation}")

    if os.path.exists(FOLDX_REPAIR_PDB):
        st.success("✅ FoldX repaired structure available")
        with open(FOLDX_REPAIR_PDB, "rb") as f:
            st.download_button(
                "⬇️ Download FoldX Repaired Structure",
                f.read(),
                file_name="1cei_Repair.pdb",
                mime="chemical/x-pdb",
                key="foldx_repaired_download"
            )

        if active_mutation:
            # FoldX BuildModel uses an individual-list file such as A F88V.
            foldx_line = f"A {active_mutation}\n"
            st.markdown("**Mutation-specific FoldX input**")
            st.code(foldx_line, language="text")
            st.download_button(
                "⬇️ Download FoldX individual_list.txt",
                foldx_line,
                file_name=f"individual_list_{active_mutation}.txt",
                mime="text/plain",
                key=f"foldx_mutation_download_{active_mutation}"
            )
            st.info(
                "This prepares the exact mutation input for FoldX BuildModel. "
                "A live FoldX calculation requires the licensed FoldX executable; "
                "the public Streamlit deployment does not contain that executable."
            )
    else:
        st.warning("FoldX repaired structure not found.")

    if os.path.exists(FOLDX_REPAIR_FXOUT):
        st.download_button(
            "⬇️ Download FoldX Repair Output",
            Path(FOLDX_REPAIR_FXOUT).read_bytes(),
            file_name="1cei_Repair.fxout",
            mime="text/plain",
            key="foldx_fxout_download"
        )

    st.divider()

    # --------------------------------------------------------
    # DYNAMUT2
    # --------------------------------------------------------
    st.subheader(f"🧬 DynaMut2 — {active_mutation}")

    if os.path.exists(DYNAMUT2_PDB):
        st.success("✅ DynaMut2 input structure available")

        with open(DYNAMUT2_PDB, "rb") as f:
            st.download_button(
                "⬇️ Download DynaMut2 Structure",
                f.read(),
                file_name="1cei_DynaMut2.pdb",
                mime="chemical/x-pdb",
                key="dynamut_pdb_download"
            )

        if active_mutation:
            st.write(
                "Submit the selected point mutation to the official DynaMut2 API "
                "using chain A."
            )

            if st.button(
                f"🧬 Run DynaMut2 for {active_mutation}",
                use_container_width=True,
                key=f"dynamut_run_{active_mutation}"
            ):
                try:
                    import requests

                    with st.spinner("Submitting mutation to DynaMut2..."):
                        with open(DYNAMUT2_PDB, "rb") as pdb_file:
                            response = requests.post(
                                "https://biosig.lab.uq.edu.au/dynamut2/api/prediction_single",
                                files={"pdb_file": ("1cei_DynaMut2.pdb", pdb_file, "chemical/x-pdb")},
                                data={"chain": "A", "mutation": active_mutation},
                                timeout=60,
                            )

                    response.raise_for_status()
                    submission = response.json()
                    job_id = submission.get("job_id")

                    if not job_id:
                        st.error(f"DynaMut2 did not return a job ID: {submission}")
                    else:
                        st.info(f"DynaMut2 job submitted: `{job_id}`")

                        result = None
                        with st.spinner("Waiting for DynaMut2 to finish..."):
                            for _ in range(30):
                                time.sleep(5)
                                poll = requests.get(
                                    "https://biosig.lab.uq.edu.au/dynamut2/api/prediction_single",
                                    params={"job_id": job_id},
                                    timeout=30,
                                )
                                poll.raise_for_status()
                                data = poll.json()
                                if data.get("message") != "RUNNING":
                                    result = data
                                    break

                        if result and result.get("prediction") is not None:
                            st.session_state[f"dynamut_result_{active_mutation}"] = result
                            st.rerun()
                        else:
                            st.warning(
                                "DynaMut2 is still processing or did not return a result within the wait window. "
                                f"Job ID: {job_id}"
                            )
                            st.markdown(
                                f"[Open DynaMut2 results](https://biosig.lab.uq.edu.au/dynamut2/results_prediction/{job_id})"
                            )

                except Exception as e:
                    st.error("DynaMut2 submission failed.")
                    st.code(str(e))

            result = st.session_state.get(f"dynamut_result_{active_mutation}")
            if result:
                st.success(f"✅ DynaMut2 result available for {active_mutation}")
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.metric("Predicted ΔΔG", f"{result.get('prediction', 'N/A')} kcal/mol")
                with d2:
                    st.metric("Chain", str(result.get("chain", "A")))
                with d3:
                    st.metric("Residue", str(result.get("res_number", int(position))))
                if result.get("results_page"):
                    st.markdown(f"[Open detailed DynaMut2 result page]({result['results_page']})")
    else:
        st.warning("DynaMut2 structure not found.")

    st.divider()

    # --------------------------------------------------------
    # ALPHAFOLD
    # --------------------------------------------------------
    st.subheader(f"🔬 AlphaFold Input — {active_mutation}")

    if os.path.exists(ALPHAFOLD_ZIP):
        st.success("✅ AlphaFold integration files available")

        try:
            with zipfile.ZipFile(ALPHAFOLD_ZIP, "r") as z:
                names = z.namelist()
                fasta_name = next((n for n in names if n.endswith("wild_type.fasta")), None)
                status_name = next((n for n in names if n.endswith("alphafold_status.json")), None)

                if status_name:
                    try:
                        status_data = json.loads(z.read(status_name).decode("utf-8"))
                        st.markdown("### 🔬 AlphaFold Integration Status")
                        st.json(status_data)
                    except Exception:
                        pass

                if fasta_name:
                    wild_fasta = z.read(fasta_name).decode("utf-8").strip()
                    lines = wild_fasta.splitlines()
                    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))

                    if active_mutation:
                        m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", active_mutation)
                        if m:
                            wt, pos, mt = m.group(1), int(m.group(2)), m.group(3)
                            if 1 <= pos <= len(sequence) and sequence[pos - 1] == wt:
                                mutant_sequence = sequence[:pos - 1] + mt + sequence[pos:]
                                mutant_fasta = f">MutAI_{active_mutation}\n{mutant_sequence}\n"
                                st.markdown(f"### 🧬 Mutation-specific FASTA — {active_mutation}")
                                st.code(mutant_fasta, language="text")
                                st.download_button(
                                    "⬇️ Download Mutation FASTA",
                                    mutant_fasta,
                                    file_name=f"{active_mutation}.fasta",
                                    mime="text/plain",
                                    key=f"af_fasta_{active_mutation}"
                                )
                            else:
                                st.warning(
                                    f"The selected mutation {active_mutation} does not match the wild-type FASTA at that position. "
                                    "No mutant FASTA was generated."
                                )
                    else:
                        st.markdown("### 🧬 Wild-Type Protein Sequence")
                        st.code(wild_fasta, language="text")

                st.download_button(
                    "⬇️ Download AlphaFold Integration Files",
                    Path(ALPHAFOLD_ZIP).read_bytes(),
                    file_name="AlphaFold_MutAI.zip",
                    mime="application/zip",
                    key="af_zip_download"
                )

        except Exception as e:
            st.error("⚠️ Could not read AlphaFold integration files.")
            st.code(str(e))
    else:
        st.warning("⚠️ AlphaFold integration files were not found.")

# ============================================================
# FOOTER
# ============================================================

st.caption(
    "🧬 MutAI • Protein Mutation Analysis Platform • Research Prototype"
)
