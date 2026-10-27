"""Streamlit front end for the persisted Mobile Price Prediction pipeline.

Run from any directory: streamlit run /path/to/app.py
The notebook recreates this source, mobile_pipeline.py and the model artifacts.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from mobile_pipeline import predict_batch, validate_inputs


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MAX_BATCH_ROWS = 10_000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

LABELS = {
    "battery_power": "Battery capacity (mAh)",
    "blue": "Bluetooth",
    "clock_speed": "Processor clock (GHz)",
    "dual_sim": "Dual SIM",
    "fc": "Front camera (MP)",
    "four_g": "4G support",
    "int_memory": "Internal storage (GB)",
    "m_dep": "Phone depth (cm)",
    "mobile_wt": "Weight category",
    "n_cores": "Processor cores",
    "pc": "Primary camera (MP)",
    "px_height": "Display height (pixels)",
    "px_width": "Display width (pixels)",
    "ram": "RAM (MB)",
    "sc_h": "Screen height (cm)",
    "sc_w": "Screen width (cm)",
    "talk_time": "Talk time (hours)",
    "three_g": "3G support",
    "touch_screen": "Touchscreen",
    "wifi": "Wi-Fi",
}
GROUPS = {
    "Performance": ["ram", "clock_speed", "n_cores", "int_memory"],
    "Display": ["px_height", "px_width", "sc_h", "sc_w"],
    "Camera": ["pc", "fc"],
    "Connectivity": ["blue", "dual_sim", "three_g", "four_g", "touch_screen", "wifi"],
    "Hardware & battery": ["battery_power", "talk_time", "m_dep", "mobile_wt"],
}


@st.cache_resource(show_spinner="Loading the fitted model…")
def load_assets(artifact_dir: str = str(ARTIFACTS)):
    """Load the complete trusted local pipeline once per artifact directory."""
    folder = Path(artifact_dir)
    missing = [name for name in ("model.joblib", "schema.json") if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing " + ", ".join(missing))
    schema = json.loads((folder / "schema.json").read_text(encoding="utf-8"))
    if not {"feature_order", "features", "class_labels"}.issubset(schema):
        raise ValueError("The input schema is incomplete. Regenerate the notebook artifacts.")
    if len(schema["feature_order"]) != 20 or set(schema["feature_order"]) != set(LABELS):
        raise ValueError("The input schema does not match the twenty expected predictors.")
    model = joblib.load(folder / "model.joblib")
    return model, schema


def default_inputs(schema: dict) -> dict:
    return {name: schema["features"][name]["default"] for name in schema["feature_order"]}


def score_inputs(model, schema: dict, values: dict) -> pd.DataFrame:
    """Use exactly the persisted raw feature order and common validation path."""
    frame = pd.DataFrame([values], columns=schema["feature_order"])
    clean = validate_inputs(frame, schema)
    return predict_batch(model, clean, schema)


def score_csv(model, schema: dict, content: bytes) -> pd.DataFrame:
    """Validate uploaded bytes and preserve input row order and optional IDs."""
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload a CSV smaller than 5 MB.")
    try:
        # IDs are opaque: preserve leading zeros and strings such as "NA".
        # Empty predictors still fail the shared numeric/category validation.
        frame = pd.read_csv(io.BytesIO(content), dtype={"id": "string"}, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError("This file could not be read as a UTF-8 CSV with a header row.") from exc
    if frame.empty:
        raise ValueError("The CSV contains no data rows.")
    if len(frame) > MAX_BATCH_ROWS:
        raise ValueError(f"Upload at most {MAX_BATCH_ROWS:,} rows at a time.")
    return predict_batch(model, frame, schema)


def compare_scenario(model, schema: dict, baseline: dict, ram: int, battery_power: int):
    """Compare two valid observations without mutating the stored baseline."""
    modified = {**baseline, "ram": ram, "battery_power": battery_power}
    return score_inputs(model, schema, baseline), score_inputs(model, schema, modified)


def class_label(schema: dict, value: int) -> str:
    return str(schema["class_labels"].get(str(value), f"Class {value}"))


def probability_table(result: pd.DataFrame, schema: dict) -> pd.DataFrame:
    columns = [f"p_{label}" for label in range(4)]
    if not all(column in result for column in columns):
        return pd.DataFrame()
    return pd.DataFrame(
        {"Price category": [f"{label} · {class_label(schema, label)}" for label in range(4)],
         "Probability": [float(result.iloc[0][column]) for column in columns]}
    ).set_index("Price category")


def input_widget(name: str, schema: dict, *, key_prefix: str = "spec_"):
    spec = schema["features"][name]
    value = spec["default"]
    key = key_prefix + name
    label = LABELS[name]
    if spec["dtype"] == "category":
        options = list(spec["categories"])
        return st.selectbox(label, options, index=options.index(value), key=key)
    if name in {"four_g", "three_g"}:
        return st.selectbox(
            label, [0, 1], index=int(value),
            format_func=lambda flag: "Yes" if flag else "No", key=key,
        )
    is_integer = spec["dtype"] == "int"
    cast = int if is_integer else float
    return st.number_input(
        label,
        min_value=cast(spec["min"]), max_value=cast(spec["max"]), value=cast(value),
        step=1 if is_integer else 0.1,
        format="%d" if is_integer else "%.1f",
        help=f"Observed training range: {spec['min']}–{spec['max']}. "
             + ("Recorded zero values are supported." if float(spec["min"]) == 0 else ""),
        key=key,
    )


def render_result(result: pd.DataFrame, schema: dict, *, prefix: str = "Predicted"):
    value = int(result.iloc[0]["price_range"])
    st.metric(f"{prefix} relative price category", f"{value} · {class_label(schema, value)}")
    probabilities = probability_table(result, schema)
    if not probabilities.empty:
        st.bar_chart(probabilities, color="#14b8a6", horizontal=True, height=225)
        st.dataframe(probabilities.style.format("{:.1%}"), width="stretch")
        st.caption("These are model probability estimates, not guaranteed correctness or market prices.")


def main():
    st.set_page_config(
        page_title="Mobile Price Lab", page_icon="📱", layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """<style>
        .block-container {padding-top: 2rem; max-width: 1240px;}
        [data-testid="stMetric"] {background: #eff6f8; border-radius: 12px;
          padding: 16px 20px; border: 1px solid #dbe8ee;}
        h1 {letter-spacing: -0.04em;}
        div[data-testid="stTabs"] button {font-size: 1rem;}
        </style>""", unsafe_allow_html=True,
    )
    st.caption("APPLIED MACHINE LEARNING · BITS PILANI WILP")
    st.title("Mobile Price Lab")
    st.write("Explore how a phone’s specifications relate to its predicted price category.")
    try:
        model, schema = load_assets()
    except Exception as exc:
        st.error(f"The model is not ready: {exc}")
        st.info("Run the submission notebook from top to bottom to recreate artifacts/model.joblib "
                "and artifacts/schema.json beside app.py, then restart the app.")
        st.stop()

    defaults = default_inputs(schema)
    if "baseline_inputs" not in st.session_state:
        st.session_state.baseline_inputs = defaults.copy()
    for state_key, feature in (("whatif_ram", "ram"), ("whatif_battery", "battery_power")):
        if state_key not in st.session_state:
            st.session_state[state_key] = int(st.session_state.baseline_inputs[feature])
    with st.sidebar:
        st.subheader("A specification-to-category model")
        st.write("Classes **0–3** describe relative price ranges in the supplied dataset. "
                 "They do not correspond to verified currency amounts.")
        st.markdown("**Selected model**")
        st.write(schema.get("model_name", "Persisted classification pipeline"))
        if "holdout_macro_f1" in schema:
            st.metric("Labeled holdout macro-F1", f"{schema['holdout_macro_f1']:.3f}")
        st.caption("Holdout results were measured before refitting on all labeled examples.")
        st.divider()
        st.caption("Start with the training medians and common categories; adjust specifications, "
                   "predict, then explore RAM and battery changes in What-if.")

    single_tab, whatif_tab, batch_tab, model_tab = st.tabs(
        ["Single phone", "What-if simulator", "Batch CSV", "Model card"]
    )
    with single_tab:
        st.subheader("Build a phone profile")
        st.caption("All twenty raw predictors are required. Defaults and input ranges come from "
                   "the training data. Zeros observed in the dataset remain valid.")
        values = {}
        with st.form("phone_form"):
            for group, names in GROUPS.items():
                st.markdown(f"#### {group}")
                columns = st.columns(3)
                for index, name in enumerate(names):
                    with columns[index % 3]:
                        values[name] = input_widget(name, schema)
            submitted = st.form_submit_button("Predict price category", type="primary", key="predict_single")
        if submitted:
            try:
                prediction = score_inputs(model, schema, values)
            except ValueError as exc:
                st.error(f"Please check the specifications: {exc}")
            else:
                st.session_state.baseline_inputs = values.copy()
                st.session_state.single_prediction = prediction
                st.session_state.whatif_ram = int(values["ram"])
                st.session_state.whatif_battery = int(values["battery_power"])
                st.session_state.pop("scenario_results", None)
        if "single_prediction" in st.session_state:
            st.divider()
            render_result(st.session_state.single_prediction, schema)
            st.caption("Result uses the most recently submitted profile; submit again after editing fields.")

    with whatif_tab:
        st.subheader("Change two specifications. Compare the result.")
        st.write("The baseline is your most recently predicted phone, or the default profile before "
                 "the first prediction. All other specifications stay fixed.")
        baseline = st.session_state.baseline_inputs
        st.caption(f"Baseline: {int(baseline['ram']):,} MB RAM · "
                   f"{int(baseline['battery_power']):,} mAh battery")
        with st.form("scenario_form"):
            left, right = st.columns(2)
            with left:
                ram_spec = schema["features"]["ram"]
                ram = st.slider("Scenario RAM (MB)", min_value=int(ram_spec["min"]),
                                max_value=int(ram_spec["max"]),
                                step=1, key="whatif_ram")
            with right:
                battery_spec = schema["features"]["battery_power"]
                battery = st.slider("Scenario battery (mAh)", min_value=int(battery_spec["min"]),
                                    max_value=int(battery_spec["max"]),
                                    step=1, key="whatif_battery")
            compare = st.form_submit_button("Compare scenarios", type="primary", key="compare_scenarios")
        if compare:
            try:
                results = compare_scenario(model, schema, baseline, ram, battery)
            except ValueError as exc:
                st.error(f"Please check the scenario: {exc}")
            else:
                st.session_state.scenario_results = results
                st.session_state.scenario_values = (ram, battery)
        if "scenario_results" in st.session_state:
            original, modified = st.session_state.scenario_results
            left, right = st.columns(2)
            with left:
                render_result(original, schema, prefix="Baseline")
            with right:
                render_result(modified, schema, prefix="Scenario")
            used_ram, used_battery = st.session_state.scenario_values
            st.caption(f"Compared scenario: {used_ram:,} MB RAM and {used_battery:,} mAh battery. "
                       "Press Compare scenarios after further edits.")
        st.info("This is a predictive association, not a causal experiment. Changing hardware "
                "does not establish a real-world price change. Some combinations may be uncommon "
                "in the training data.")

    with batch_tab:
        st.subheader("Score a CSV of phone specifications")
        st.write("Provide the same twenty raw columns. An optional `id` is retained in order; "
                 "known `Unnamed:` export columns are removed. Targets, missing predictors, "
                 "unexpected columns and invalid values are rejected with a clear message.")
        template = pd.DataFrame([defaults], columns=schema["feature_order"]).to_csv(index=False)
        st.download_button("Download input template", template, "phone_input_template.csv", "text/csv")
        upload = st.file_uploader("Upload phone specifications", type=["csv"], key="batch_upload",
                                  help="UTF-8 CSV, up to 5 MB and 10,000 rows; no target column.")
        if upload is not None:
            try:
                predictions = score_csv(model, schema, upload.getvalue())
            except ValueError as exc:
                st.error(f"The batch could not be scored: {exc}")
            else:
                st.success(f"Scored {len(predictions):,} phones. Original row order is preserved.")
                st.dataframe(predictions.head(50), width="stretch", hide_index=True)
                st.caption("Preview shows the first 50 rows. The download contains every scored row.")
                st.download_button("Download predictions", predictions.to_csv(index=False),
                                   "mobile_price_predictions.csv", "text/csv", key="download_predictions")

    with model_tab:
        st.subheader("Evidence and limits")
        metrics = st.columns(3)
        for column, label, key in zip(metrics,
                ["Development CV macro-F1", "Labeled holdout accuracy", "Labeled holdout macro-F1"],
                ["cv_macro_f1", "holdout_accuracy", "holdout_macro_f1"]):
            with column:
                value = schema.get(key)
                st.metric(label, f"{value:.3f}" if isinstance(value, (int, float)) else "See notebook")
        st.write(f"**Deployment training rows:** {schema.get('trained_rows', 'See notebook'):,}"
                 if isinstance(schema.get("trained_rows"), int) else "**Training details:** see notebook.")
        st.write("The model configuration was selected using stratified development cross-validation. "
                 "The supplied test file has no labels: its predictions have no measured test accuracy. "
                 "The final deployment model is refitted on all 2,000 labeled observations after "
                 "the holdout evaluation was recorded.")
        st.write("The saved pipeline contains the fitted preprocessing and classifier. The application "
                 "uses the same schema, category encoding and engineered features as the notebook. "
                 "No identifier is used as a predictor.")
        st.warning("The dataset supports an academic classification exercise. Its provenance, "
                   "currency ranges and relevance to today’s devices are not established. "
                   "Probability estimates are not a promise of correctness.")
        st.markdown("**Class legend**")
        st.dataframe(pd.DataFrame({"price_range": list(range(4)),
                                  "Relative category": [class_label(schema, i) for i in range(4)]}),
                     hide_index=True, width="stretch")
        st.caption("Verified local execution is documented in the notebook. A public hosted URL "
                   "is a separate deployment step and is not implied by this interface.")


if __name__ == "__main__":
    main()
