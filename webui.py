"""
Streamlit WebUI for SF-T (SimpleFold-Turbo) - Fast Protein Structure Prediction

SF-T achieves 9-14x speedup over baseline diffusion models using
timestep-aware caching (TeaCache), enabling rapid ensemble generation.

G Taghon
2026
"""

import sys
import time
import logging
import tempfile
from pathlib import Path

# Get artifacts directory (contains model checkpoints and cache)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
from typing import List
from argparse import Namespace

# Add simplefold package path for internal imports (same as cli.py)
sys.path.append(str(Path(__file__).resolve().parent / "src" / "simplefold"))

import numpy as np
import torch
import streamlit as st
import streamlit.components.v1 as components

from simplefold.inference import predict_structures_from_fastas

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="SF-T 0.1",
    page_icon="𓇦",
    layout="wide"
)

def get_device() -> str:
    """Determine the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def validate_protein_sequence(seq: str) -> tuple[bool, str]:
    """
    Validate a protein sequence.

    Returns:
        tuple: (is_valid, cleaned_sequence or error_message)
    """
    # Clean sequence
    seq = seq.strip().replace(' ', '').replace('\n', '').upper()

    # Check for empty
    if not seq:
        return False, "Please enter a protein sequence."

    # Valid amino acids
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    invalid_chars = set(seq) - valid_aa

    if invalid_chars:
        return False, f"Invalid amino acid characters: {', '.join(sorted(invalid_chars))}"

    # Length check
    if len(seq) < 10:
        return False, "Sequence too short (minimum 10 residues)."
    if len(seq) > 512:
        return False, "Sequence too long (maximum 512 residues for web interface)."

    return True, seq

def get_model_name(model_size: str) -> str:
    """Convert UI model size to internal model name."""
    model_map = {
        "100M": "simplefold_100M",
        "360M": "simplefold_360M",
        "700M": "simplefold_700M",
        "1.1B": "simplefold_1.1B",
        "1.6B": "simplefold_1.6B",
        "3B": "simplefold_3B",
    }
    return model_map.get(model_size, "simplefold_100M")

def fold_sequence(
    sequence: str,
    model_size: str,
    ensemble_size: int,
    use_teacache: bool,
    threshold: float,
    progress_callback=None
) -> List[str]:
    """
    Fold a protein sequence and return PDB strings.

    Returns:
        List of PDB format strings, one per ensemble member.
    """
    # Create temporary directory for this folding job
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write sequence to FASTA file
        fasta_path = tmpdir / "input.fasta"
        with open(fasta_path, 'w') as f:
            f.write(f">query|protein\n{sequence}\n")

        # Set up args for predict_structures_from_fastas
        args = Namespace(
            simplefold_model=get_model_name(model_size),
            ckpt_dir=str(ARTIFACTS_DIR),  # Use local artifacts for model checkpoints
            cache_dir=str(CACHE_DIR),
            output_dir=str(output_dir),
            num_steps=500,
            tau=0.1,
            no_log_timesteps=False,
            fasta_path=str(fasta_path),
            nsample_per_protein=ensemble_size,
            plddt=False,
            output_format="pdb",
            backend="torch",
            teacache=threshold if use_teacache else 0.0,
            seed=42,
        )

        # Run folding
        if progress_callback:
            progress_callback(0, ensemble_size)

        predict_structures_from_fastas(args)

        # Collect output PDB files
        pdb_strings = []
        prediction_dir = output_dir / f"predictions_{args.simplefold_model}"

        # Find all PDB files
        pdb_files = sorted(prediction_dir.glob("**/*.pdb"))
        logger.info(f"Found {len(pdb_files)} PDB files in {prediction_dir}")

        if not pdb_files:
            # Check what's in the output directory
            all_files = list(output_dir.rglob("*"))
            logger.warning(f"No PDB files found. Output dir contents: {[str(f) for f in all_files[:20]]}")

        for i, pdb_file in enumerate(pdb_files):
            content = pdb_file.read_text()
            if content.strip():
                pdb_strings.append(content)
            else:
                logger.warning(f"Empty PDB file: {pdb_file}")
            if progress_callback:
                progress_callback(i + 1, len(pdb_files))

        return pdb_strings


def extract_ca_coords(pdb_string: str) -> np.ndarray:
    """Extract C-alpha coordinates from a PDB string."""
    coords = []
    for line in pdb_string.split('\n'):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
    return np.array(coords)


def kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> tuple:
    """
    Compute optimal rotation matrix to align mobile onto reference using Kabsch algorithm.
    Returns (rotation_matrix, translation_vector, centroid_ref, centroid_mobile).
    """
    # Center both structures
    centroid_mobile = mobile.mean(axis=0)
    centroid_ref = reference.mean(axis=0)

    mobile_centered = mobile - centroid_mobile
    ref_centered = reference - centroid_ref

    # Compute covariance matrix
    H = mobile_centered.T @ ref_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R, centroid_ref, centroid_mobile


def transform_pdb(pdb_string: str, R: np.ndarray, centroid_ref: np.ndarray, centroid_mobile: np.ndarray) -> str:
    """Apply rotation and translation to all ATOM coordinates in a PDB string."""
    lines = []
    for line in pdb_string.split('\n'):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            # Apply transformation: R @ (coord - centroid_mobile) + centroid_ref
            coord = np.array([x, y, z])
            new_coord = R @ (coord - centroid_mobile) + centroid_ref

            # Reconstruct line with new coordinates
            new_line = line[:30] + f"{new_coord[0]:8.3f}{new_coord[1]:8.3f}{new_coord[2]:8.3f}" + line[54:]
            lines.append(new_line)
        else:
            lines.append(line)
    return '\n'.join(lines)


def align_ensemble(pdb_list: List[str]) -> List[str]:
    """Align all structures in the ensemble to the first structure using C-alpha atoms."""
    if len(pdb_list) <= 1:
        return pdb_list

    # Extract reference CA coords
    ref_coords = extract_ca_coords(pdb_list[0])
    if len(ref_coords) == 0:
        logger.warning("No CA atoms found in reference structure")
        return pdb_list

    aligned = [pdb_list[0]]  # Reference stays unchanged

    for pdb in pdb_list[1:]:
        mobile_coords = extract_ca_coords(pdb)

        if len(mobile_coords) == len(ref_coords):
            R, centroid_ref, centroid_mobile = kabsch_align(mobile_coords, ref_coords)
            aligned_pdb = transform_pdb(pdb, R, centroid_ref, centroid_mobile)
            aligned.append(aligned_pdb)
        else:
            logger.warning(f"CA atom count mismatch: {len(mobile_coords)} vs {len(ref_coords)}")
            aligned.append(pdb)

    return aligned


def hsl_to_hex(h: int, s: int, l: int) -> str:
    """Convert HSL to hex color string."""
    s = s / 100
    l = l / 100
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2

    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r = int((r + m) * 255)
    g = int((g + m) * 255)
    b = int((b + m) * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def create_ensemble_viewer(pdb_list: List[str], height: int = 500) -> str:
    """Create HTML for 3Dmol.js viewer showing all ensemble members aligned."""
    import html
    import json

    # Generate distinct hex colors for each model
    n_models = len(pdb_list)
    colors = [hsl_to_hex(int(i * 360 / n_models), 70, 50) for i in range(n_models)]

    # Store PDB data in hidden textareas and colors as JSON
    pdb_textareas = "\n".join([
        f'<textarea id="pdb_data_{i}" style="display:none;">{html.escape(pdb)}</textarea>'
        for i, pdb in enumerate(pdb_list)
    ])
    colors_json = json.dumps(colors)

    return f"""
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <div id="ensemble_viewer" style="height: {height}px; width: 100%; position: relative; border: 1px solid #ddd; border-radius: 4px;"></div>
    {pdb_textareas}
    <script>
        (function() {{
            var element = document.getElementById('ensemble_viewer');
            var config = {{ backgroundColor: 'white' }};
            var viewer = $3Dmol.createViewer(element, config);

            // Hex colors for each model
            var colors = {colors_json};
            var numModels = {n_models};

            // Add all models from hidden textareas
            for (var i = 0; i < numModels; i++) {{
                var pdbData = document.getElementById('pdb_data_' + i).value;
                viewer.addModel(pdbData, "pdb");
            }}

            // Style each model with its unique color
            for (var i = 0; i < numModels; i++) {{
                viewer.setStyle({{model: i}}, {{cartoon: {{color: colors[i], opacity: 0.85}}}});
            }}

            viewer.zoomTo();
            viewer.render();
        }})();
    </script>
    """


def create_single_viewer(pdb_data: str, height: int = 500) -> str:
    """Create HTML for 3Dmol.js viewer showing a single structure."""
    import html
    pdb_escaped = html.escape(pdb_data)

    return f"""
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <div id="viewer_single" style="height: {height}px; width: 100%; position: relative; border: 1px solid #ddd; border-radius: 4px;"></div>
    <textarea id="pdb_data_single" style="display:none;">{pdb_escaped}</textarea>
    <script>
        (function() {{
            var element = document.getElementById('viewer_single');
            var config = {{ backgroundColor: 'white' }};
            var viewer = $3Dmol.createViewer(element, config);
            var pdbData = document.getElementById('pdb_data_single').value;

            viewer.addModel(pdbData, "pdb");
            viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
            viewer.zoomTo();
            viewer.render();
        }})();
    </script>
    """

def main():
    """Main Streamlit application."""

    # Header
    st.title("𓇦SF-T")
    st.markdown("""
    **Fast protein structure prediction with ensemble generation.**

    SF-T uses timestep-aware caching to achieve **9-14x speedup** over standard
    diffusion models, enabling rapid generation of structural ensembles.
    """)

    # Sidebar for parameters
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model selection
        model_size = st.selectbox(
            "Model Size",
            options=["100M", "360M", "700M", "1.1B", "1.6B", "3B"],
            index=0,
            help="Larger models are more accurate but slower. 100M recommended for quick exploration."
        )

        # Ensemble size - highlight the speed advantage
        ensemble_size = st.slider(
            "Ensemble Size",
            min_value=1,
            max_value=20,
            value=5,
            help="Generate multiple structures to explore conformational diversity. SF-T can generate 10+ structures in the time traditional methods take for one!"
        )

        st.divider()

        # TeaCache settings
        st.subheader("⏩ TeaCache Acceleration")
        use_teacache = st.checkbox(
            "Enable TeaCache",
            value=True,
            help="Enable timestep-aware caching for faster generation."
        )

        if use_teacache:
            threshold = st.slider(
                "Cache Threshold (θ)",
                min_value=0.05,
                max_value=0.5,
                value=0.1,
                step=0.05,
                help="Lower = higher quality, slower. Higher = faster, slight quality reduction. Default 0.1 is optimal."
            )
        else:
            threshold = 0.0

        st.divider()

        # Device info
        device = get_device()
        st.caption(f"**Device:** {device.upper()}")

        # Speed estimate
        if use_teacache:
            est_time = ensemble_size * 1.0  # ~1s per structure with TeaCache
            st.caption(f"**Est. time:** ~{est_time:.0f}s for {ensemble_size} structures")
        else:
            est_time = ensemble_size * 10.0  # ~10s per structure without
            st.caption(f"**Est. time:** ~{est_time:.0f}s for {ensemble_size} structures")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Input Sequence")

        # Sequence input
        sequence_input = st.text_area(
            "Protein Sequence",
            height=150,
            placeholder="Enter amino acid sequence (e.g., MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQLTLQ...)",
            help="Standard single-letter amino acid codes. Max 512 residues for web interface."
        )

        # Example sequences
        with st.expander("📋 Example Sequences"):
            examples = {
                "Insulin (51 aa)": "GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVCGERGFFYTPKT",
                "Ubiquitin (76 aa)": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
                "GFP (238 aa)": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
            }

            for name, seq in examples.items():
                if st.button(f"Use {name}", key=f"ex_{name}"):
                    st.session_state.sequence = seq
                    st.rerun()

        # Use session state for sequence
        if 'sequence' in st.session_state:
            sequence_input = st.session_state.sequence

        # Fold button
        fold_button = st.button("🔬 Fold Sequence", type="primary", use_container_width=True)

    with col2:
        st.subheader("🧬 Results")

        if fold_button and sequence_input:
            # Validate sequence
            is_valid, result = validate_protein_sequence(sequence_input)

            if not is_valid:
                st.error(result)
            else:
                sequence = result
                st.info(f"Folding {len(sequence)}-residue sequence ({ensemble_size} structures)...")

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    start_time = time.time()

                    def update_progress(current, total):
                        # current is already 1-indexed from fold_sequence
                        progress = min(current / total, 1.0)  # Clamp to avoid overflow
                        progress_bar.progress(progress)
                        status_text.text(f"Generating structure {current}/{total}...")

                    # Fold
                    pdb_strings = fold_sequence(
                        sequence=sequence,
                        model_size=model_size,
                        ensemble_size=ensemble_size,
                        use_teacache=use_teacache,
                        threshold=threshold,
                        progress_callback=update_progress
                    )

                    elapsed = time.time() - start_time
                    progress_bar.progress(1.0)
                    status_text.text(f"✅ Generated {ensemble_size} structures in {elapsed:.1f}s ({elapsed/ensemble_size:.2f}s/structure)")

                    # Store results
                    st.session_state.pdb_results = pdb_strings
                    st.session_state.fold_time = elapsed

                except Exception as e:
                    st.error(f"Folding failed: {str(e)}")
                    logger.exception("Folding error")

        # Display results
        if 'pdb_results' in st.session_state and st.session_state.pdb_results:
            pdb_strings = st.session_state.pdb_results

            # View mode selection
            if len(pdb_strings) > 1:
                view_col1, view_col2 = st.columns([1, 2])
                with view_col1:
                    view_mode = st.radio(
                        "View Mode",
                        ["Ensemble", "Single"],
                        horizontal=True,
                        help="Ensemble shows all structures aligned (NMR-style). Single shows one at a time."
                    )
                with view_col2:
                    if view_mode == "Single":
                        model_idx = st.selectbox(
                            "Select Model",
                            options=range(len(pdb_strings)),
                            format_func=lambda x: f"Model {x + 1}",
                            key="model_viewer_select"
                        )
                    else:
                        st.caption(f"Showing {len(pdb_strings)} aligned structures")
            else:
                view_mode = "Single"
                model_idx = 0

            # 3D viewer
            if view_mode == "Ensemble" and len(pdb_strings) > 1:
                # Align structures by C-alpha atoms before displaying
                aligned_pdbs = align_ensemble(pdb_strings)
                components.html(
                    create_ensemble_viewer(aligned_pdbs),
                    height=520
                )
            else:
                components.html(
                    create_single_viewer(pdb_strings[model_idx]),
                    height=520
                )

            # Download buttons
            st.divider()

            if len(pdb_strings) > 1:
                # Multiple structures - show zip download prominently
                import io
                import zipfile

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for i, pdb in enumerate(pdb_strings):
                        zf.writestr(f"sft_model_{i+1}.pdb", pdb)

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        f"📦 Download Ensemble ({len(pdb_strings)} structures)",
                        zip_buffer.getvalue(),
                        "sft_ensemble.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                with dl_col2:
                    # Individual model download (default to first if in ensemble mode)
                    selected_idx = model_idx if view_mode == "Single" else 0
                    st.download_button(
                        f"📥 Download Model {selected_idx + 1}",
                        pdb_strings[selected_idx],
                        f"sft_model_{selected_idx + 1}.pdb",
                        mime="chemical/x-pdb",
                        use_container_width=True
                    )
            else:
                # Single structure
                st.download_button(
                    "📥 Download Structure",
                    pdb_strings[0],
                    "sft_prediction.pdb",
                    mime="chemical/x-pdb",
                    use_container_width=True
                )

    # Footer
    st.divider()
    st.caption("""
    **SF-T** | Accelerated protein structure prediction using TeaCache
    For research use only. Not for clinical or diagnostic purposes.
    """)

if __name__ == "__main__":
    main()
