
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import zipfile
import tempfile
import subprocess
import shutil
import time
import re
import requests

# ============================================================
# MUTATION-SPECIFIC STRUCTURAL ANALYSIS
# ============================================================

st.header("🧬 Structural Analysis")

if "last_mutation" not in st.session_state:
    st.info(
        "Enter a mutation above and click **Predict Mutation & Prepare "
        "Structural Analysis**. FoldX, DynaMut2 and AlphaFold inputs will then "
        "be tied to that selected mutation."
    )
else:
    active_mutation = st.session_state["last_mutation"]

    st.subheader(f"🧪 FoldX — {active_mutation}")

    if os.path.exists(FOLDX_REPAIR_PDB):
        st.success("✅ FoldX repaired structure available")

        st.download_button(
            "⬇️ Download FoldX Repaired Structure",
            data=Path(FOLDX_REPAIR_PDB).read_bytes(),
            file_name="1cei_Repair.pdb",
            mime="chemical/x-pdb",
            key="download_foldx_repaired"
        )

        st.write("Mutation-specific FoldX input:")
        st.code(make_foldx_mutant_file(active_mutation, "A"), language="text")

        if st.button(
            f"🧪 Run FoldX BuildModel for {active_mutation}",
            use_container_width=True,
            key=f"foldx_{active_mutation}"
        ):
            with st.spinner("Running FoldX..."):
                foldx_result = run_foldx(active_mutation)

            if foldx_result["status"] == "success":
                st.success(f"✅ FoldX completed for {active_mutation}")

                dif_content = foldx_result.get("dif_content", "")
                if dif_content:
                    st.code(dif_content, language="text")

                if foldx_result.get("mutant_bytes"):
                    st.download_button(
                        "⬇️ Download FoldX Mutant PDB",
                        data=foldx_result["mutant_bytes"],
                        file_name=f"{active_mutation}_FoldX.pdb",
                        mime="chemical/x-pdb",
                        key=f"foldx_download_{active_mutation}"
                    )

            elif foldx_result["status"] == "unavailable":
                st.warning("⚠️ FoldX calculation is not available on this deployment.")
                st.info(
                    "FoldX BuildModel requires the licensed FoldX executable. "
                    "This public app cannot assume that executable is installed. "
                    "The exact mutation input above is ready for FoldX."
                )
            else:
                st.error(f"❌ FoldX failed: {foldx_result.get('message', 'Unknown error')}")
                if foldx_result.get("stderr"):
                    st.code(foldx_result["stderr"])

    else:
        st.warning("FoldX repaired structure not found.")

    if os.path.exists(FOLDX_REPAIR_FXOUT):
        st.download_button(
            "⬇️ Download FoldX Repair Output",
            data=Path(FOLDX_REPAIR_FXOUT).read_bytes(),
            file_name="1cei_Repair.fxout",
            mime="text/plain",
            key="download_foldx_fxout"
        )

    st.divider()

    # --------------------------------------------------------
    # DYNAMUT2
    # --------------------------------------------------------

    st.subheader(f"🧬 DynaMut2 — {active_mutation}")

    if os.path.exists(DYNAMUT2_PDB):
        st.success("✅ DynaMut2 input structure available")

        st.write(
            "Click the button below to submit the selected mutation to the "
            "official DynaMut2 API."
        )

        if st.button(
            f"🧬 Run DynaMut2 for {active_mutation}",
            use_container_width=True,
            key=f"dynamut_{active_mutation}"
        ):
            with st.spinner(f"Submitting {active_mutation} to DynaMut2..."):
                submission = submit_dynamut2(active_mutation)

            if submission["status"] == "submitted":
                job_id = submission["job_id"]
                st.info(f"DynaMut2 job submitted: `{job_id}`")

                with st.spinner("Waiting for DynaMut2 result..."):
                    result = poll_dynamut2(job_id)

                if result["status"] == "success":
                    st.session_state["dynamut_result"] = {
                        "mutation": active_mutation,
                        **result["data"]
                    }
                    st.rerun()
                else:
                    st.error(result.get("message", "DynaMut2 did not return a result."))
            else:
                st.error(f"DynaMut2 submission failed: {submission['message']}")

        dynamut_result = st.session_state.get("dynamut_result")

        if dynamut_result and dynamut_result.get("mutation") == active_mutation:
            st.success(f"✅ DynaMut2 result available for {active_mutation}")

            d1, d2, d3 = st.columns(3)

            with d1:
                st.metric(
                    "DynaMut2 ΔΔG",
                    str(dynamut_result.get("prediction", "N/A"))
                )
            with d2:
                st.metric(
                    "Chain",
                    str(dynamut_result.get("chain", DYNAMUT2_CHAIN))
                )
            with d3:
                st.metric(
                    "Residue",
                    str(dynamut_result.get("res_number", position))
                )

            if dynamut_result.get("results_page"):
                st.markdown(
                    f"[Open DynaMut2 result page]({dynamut_result['results_page']})"
                )

        st.download_button(
            "⬇️ Download DynaMut2 Input Structure",
            data=Path(DYNAMUT2_PDB).read_bytes(),
            file_name="1cei_DynaMut2.pdb",
            mime="chemical/x-pdb",
            key="download_dynamut_pdb"
        )

    else:
        st.warning("DynaMut2 structure not found.")

    st.divider()

    # --------------------------------------------------------
    # ALPHAFOLD
    # --------------------------------------------------------

    st.subheader(f"🔬 AlphaFold Input — {active_mutation}")

    if os.path.exists(ALPHAFOLD_ZIP):
        st.success("✅ AlphaFold integration files available")

        mutant_fasta, fasta_error = prepare_alphafold_mutant_fasta(active_mutation)

        if mutant_fasta:
            st.write("Mutation-specific FASTA prepared:")
            st.code(mutant_fasta, language="text")

            st.download_button(
                "⬇️ Download Mutation FASTA",
                data=mutant_fasta,
                file_name=f"{active_mutation}.fasta",
                mime="text/plain",
                key=f"download_af_{active_mutation}"
            )

            st.info(
                "This prepares the mutation-specific AlphaFold input. "
                "A new AlphaFold prediction is not generated by this app "
                "unless an AlphaFold runtime is connected."
            )
        else:
            st.warning(f"Could not prepare AlphaFold FASTA: {fasta_error}")

        st.download_button(
            "⬇️ Download AlphaFold Integration Files",
            data=Path(ALPHAFOLD_ZIP).read_bytes(),
            file_name="AlphaFold_MutAI.zip",
            mime="application/zip",
            key="download_af_zip"
        )
    else:
        st.warning("AlphaFold integration files were not found.")

# ============================================================
# FOOTER
# ============================================================

st.caption(
    "🧬 MutAI • Protein Mutation Analysis Platform • Research Prototype"
)
