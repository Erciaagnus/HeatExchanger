import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import shutil

# Add scripts directory to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Import refactored modules
from sampling.generate_point import generate_constrained_lhs
from sampling.mapping_to_porous import run_batch_mapping
from visualization.plot_porous_sampling import plot_porous
from surrogate.train_model import train_surrogates, save_surrogate_bundle, run_ga, load_surrogate_bundle, predict_q_dp

# Page Config
st.set_page_config(
    page_title="Heat Exchanger Design Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🔥 Heat Exchanger All-in-One Dashboard")

# Sidebar Navigation
mode = st.sidebar.radio("Navigation", [
    "Introduction",
    "1. Sampling (LHS)",
    "2. Porous Mapping",
    "3. Visualization",
    "4. Surrogate Modeling (Training)",
    "5. Optimization (GA)"
])

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
FIGURE_DIR = os.path.join(os.path.dirname(__file__), 'figure')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)



# Initialize session state for scenarios if not present
if 'scenario_params' not in st.session_state:
    st.session_state['scenario_params'] = {}

def load_scenario(name, params):
    st.session_state['scenario_params'] = params
    st.success(f"Loaded Scenario: {name}")
    st.info("Go to the relevant tabs (Sampling/Optimization) to see pre-filled values.")

if mode == "Introduction":
    st.markdown(r"""
    ### Welcome to the Heat Exchanger Design Suite
    
    This dashboard unifies the workflow for designing and optimizing fin-tube heat exchangers using porous media surrogate models.
    
    #### 📚 detailed Algorithm & Methodology
    
    **1. Latin Hypercube Sampling (LHS)**
    - **Purpose**: Efficiently sample the design space ($S_1$, $f_h$, $f_s$) with fewer points than random sampling.
    - **Constraint Handling**:
      - We impose a coupled constraint to ensure geometric feasibility: 
        $$f_h \le 0.5 \times \left( \frac{S_1}{\sqrt{2}} - D_c \right) - 0.4 \text{ mm}$$
      - This prevents fins from overlapping in a staggered arrangement.
    
    **2. Porous Media Mapping (Darcy-Forchheimer)**
    - **Physics**: We treat the fin-tube bundle as a porous medium.
    - **Equation**: The pressure drop is modeled by the Darcy-Forchheimer law:
      $$ \frac{\Delta P}{L} = \frac{\mu}{K} v + \frac{1}{2}\rho C_2 v^2 $$
      - $\mu/K$: Viscous resistance term (linear with velocity).
      - $C_2$: Inertial resistance factor (quadratic with velocity).
    - **Method**: We compute $\Delta P$ for various inlet velocities using the **Nir (1991)** correlation and fit the $v$ vs $\Delta P/L$ curve to extract $1/K$ and $C_2$.
    
    **3. Surrogate Modeling (Gaussian Process)**
    - **Goal**: Predict Heat Flux ($Q''$) and Pressure Drop ($\Delta P$) from geometry without running expensive CFD every time.
    - **Kernel**: We use a `Matern(2.5) + WhiteKernel` or `RationalQuadratic` kernel, automatically selected based on Cross-Validation ($R^2$) score.
    - **Uncertainty**: GP provides a mean prediction $\mu(x)$ and standard deviation $\sigma(x)$.
    
    **4. Genetic Algorithm (GA) Optimization**
    - **Objective**: Maximize the ratio of Heat Flux to Pressure Drop.
    - **Fitness Function**:
      - **Standard**: $F(x) = \frac{\mu_Q}{\mu_{\Delta P}}$
      - **Conservative**: $F(x) = \frac{\mu_Q - k\sigma_Q}{\mu_{\Delta P} + k\sigma_{\Delta P}}$
      - This "Conservative" approach penalizes uncertainty, guiding the optimizer towards robust designs with high confidence.
    
    ---
    
    #### 🧪 Example Scenarios
    Load a preset scenario to see how parameters affect the design.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Scenario A: Standard")
        st.write("Typical operating conditions.")
        if st.button("Load Standard Scenario"):
            load_scenario("Standard", {
                "n_samples": 100,
                "T_celsius": 14.8,
                "v_design": 2.0,
                "k_sigma": 1.0
            })
            
    with col2:
        st.subheader("Scenario B: High Precision")
        st.write("More samples, stricter safety factor.")
        if st.button("Load High Precision"):
            load_scenario("High Precision", {
                "n_samples": 5000,
                "T_celsius": 25.0,
                "v_design": 2.0,
                "k_sigma": 2.0
            })
            
    with col3:
        st.subheader("Scenario C: High Velocity")
        st.write("Higher design velocity condition.")
        if st.button("Load High Velocity"):
            load_scenario("High Velocity", {
                "n_samples": 500,
                "T_celsius": 20.0,
                "v_design": 5.0,
                "k_sigma": 1.0
            })
            
    st.image("assets/Fig1_schem.png", caption="Workflow Framework", use_column_width=True)

elif mode == "1. Sampling (LHS)":
    st.header("1. Latin Hypercube Sampling (LHS)")
    st.markdown("Generate constrained design samples.")
    
    # Defaults from session state
    defaults = st.session_state.get('scenario_params', {})

    with st.expander("ℹ️ Theoretical Background: Latin Hypercube Sampling (LHS)"):
        st.markdown(r"""
        **Latin Hypercube Sampling (LHS)** is a statistical method for generating a near-random sample of parameter values from a multidimensional distribution.
        
        - **Efficiency**: Unlike Monte Carlo (random) sampling, LHS ensures that the sample points are spread more evenly across the range of each variable.
        - **Process**:
            1. Divide the range of each variable into $N$ equal probable intervals.
            2. Place exactly one sample in each interval.
            3. Shuffle the samples to form $N$ parameter vectors.
        - **Constraints**: 
            - In this tool, we apply an additional geometric constraint:
              $$ f_h \le 0.5 \times \left( \frac{S_1}{\sqrt{2}} - D_c \right) - 0.4 \text{ mm} $$
            - This ensures that the fins do not physically overlap or collide.
        """)

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.number_input("Number of Samples", 
                                    min_value=100, 
                                    max_value=1000000, 
                                    value=defaults.get('n_samples', 100), 
                                    step=100)
    with col2:
        seed = st.number_input("Random Seed", value=2025)
        
    filename = st.text_input("Output Filename", value="constrained_LHS_samples.csv")
    
    if st.button("Generate Samples"):
        with st.spinner("Generating samples..."):
            out_path = os.path.join(DATA_DIR, filename)
            df = generate_constrained_lhs(n=n_samples, seed=seed, out_path=out_path, include_diag=True)
            st.success(f"Generated {len(df)} samples!")
            st.dataframe(df.head())
            st.download_button("Download CSV", df.to_csv(index=False), filename, "text/csv")

elif mode == "2. Porous Mapping":
    st.header("2. Map Design to Porous Parameters")
    st.markdown("Calculate porous media parameters (Inv_K, C2) from design variables.")
    
    defaults = st.session_state.get('scenario_params', {})
    
    with st.expander("ℹ️ Theoretical Background: Porous Media Mapping"):
        st.markdown(r"""
        This step converts detailed **geometric parameters** (Spacing, Height) into **macroscopic porous media parameters**.
        
        **Governing Equation (Darcy-Forchheimer Law)**:
        $$ \frac{\Delta P}{L} = \underbrace{\frac{\mu}{K} v}_{\text{Viscous Term}} + \underbrace{\frac{1}{2}\rho C_2 v^2}_{\text{Inertial Term}} $$
        
        - **Viscous Resistance ($1/K$)**: Dominates at low velocities (laminar flow).
        - **Inertial Resistance ($C_2$)**: Dominates at high velocities (turbulent flow).
        
        **Procedure**:
        1. Calculate Pressure Drop ($\Delta P$) for a range of velocities using the **Nir (1991)** correlation:
           $$ f_N = 1.1 \cdot Re^{-0.25} \cdot (S_1/D_c)^{0.6} \cdot (A_{tot}/A_{bare})^{0.15} $$
        2. Perform a regression (Curve Fitting) of $\Delta P/L$ vs. $v$ to extract $1/K$ and $C_2$.
        """)
        
    def get_files(extensions):
        files = []
        for d, label in [(DATA_DIR, 'data/'), (OUTPUT_DIR, 'output/')]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith(extensions):
                        files.append(os.path.join(label, f))
        return files

    # File selection
    files = get_files('.csv')
    selected_file_label = st.selectbox("Select Input Design File", files, index=0 if files else None)
    
    if selected_file_label:
        # Resolve full path
        if selected_file_label.startswith('data/'):
            in_path = os.path.join(DATA_DIR, selected_file_label.replace('data/', ''))
        else:
            in_path = os.path.join(OUTPUT_DIR, selected_file_label.replace('output/', ''))
            
        out_filename = st.text_input("Output Filename", value=f"porous_{os.path.basename(in_path)}")
        
        with st.expander("Advanced Configuration", expanded=True):
            T_celsius = st.number_input("Temperature (°C)", value=float(defaults.get('T_celsius', 14.8)))
            v_design = st.number_input("Design Velocity (m/s)", value=float(defaults.get('v_design', 2.019)))
            check_constraint = st.checkbox("Check Constraints", value=True)
            
        if st.button("Run Mapping"):
            with st.spinner("Calculating porous parameters (this may take a while)..."):
                out_path = os.path.join(OUTPUT_DIR, out_filename)
                
                # Progress bar callback
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # We interpret progress roughly since run_batch_mapping prints to stdout
                # Ideally we would refactor to yield progress, but for now we just run it.
                # Actually, capturing stdout in streamlit is tricky.
                # Let's just run it.
                
                try:
                    df_out = run_batch_mapping(
                        in_csv=in_path,
                        out_csv=out_path,
                        T_celsius=T_celsius,
                        v_design=v_design,
                        Dc_mm=24.0,
                        delta_f_mm=0.5,
                        pitch_ratio=1.0,
                        N=4,
                        v_min=0.5,
                        v_max=3.5,
                        n_points=15, # Reduced for speed in dashboard
                        max_rows=None,
                        progress_every=500,
                        check_constraint=check_constraint
                    )
                    progress_bar.progress(100)
                    st.success("Mapping Completed!")
                    st.dataframe(df_out.head())
                    st.download_button("Download Result CSV", df_out.to_csv(index=False), out_filename, "text/csv")
                except Exception as e:
                    st.error(f"Error during mapping: {e}")

elif mode == "3. Visualization":
    st.header("3. Visualization")
    
    with st.expander("ℹ️ visualization Guide"):
        st.markdown("""
        Visualizing the design space helps to understand the relationships between geometric inputs and porous outputs.
        
        - **3D Scatter Plot**: Shows the distribution of **Porosity**, **Viscous Resistance**, and **Inertial Resistance**.
        - **Histograms**: display the frequency distribution of each parameter, helping to check if the sampling covers the desired range uniformly.
        - **Log Scale**: Recommended for resistance values, as they often span multiple orders of magnitude.
        """)
    
    files = get_files('.csv')
    selected_file_label = st.selectbox("Select Porous Data File", files)
    
    use_log = st.checkbox("Use Log Scale", value=True)
    
    if st.button("Generate Plots"):
        if selected_file_label:
            if selected_file_label.startswith('data/'):
                in_path = os.path.join(DATA_DIR, selected_file_label.replace('data/', ''))
            else:
                in_path = os.path.join(OUTPUT_DIR, selected_file_label.replace('output/', ''))
                
            out_subdir = os.path.join(FIGURE_DIR, "dashboard_plots")
            os.makedirs(out_subdir, exist_ok=True)
            
            with st.spinner("Generating plots..."):
                try:
                    plot_porous(in_path, out_subdir, use_log=use_log)
                    st.success("Plots generated!")
                    
                    # Display plots
                    st.subheader("3D Scatter Plot")
                    suffix = "log" if use_log else "linear"
                    st.image(os.path.join(out_subdir, f"dist_3d_{suffix}.png"))
                    
                    st.subheader("Histograms")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(os.path.join(out_subdir, "hist_porosity.png"))
                    with c2:
                        st.image(os.path.join(out_subdir, f"hist_viscous_{suffix}.png"))
                    with c3:
                        st.image(os.path.join(out_subdir, f"hist_inertial_{suffix}.png"))
                        
                except Exception as e:
                    st.error(f"Error generating plots: {e}")

elif mode == "4. Surrogate Modeling (Training)":
    st.header("4. Train Surrogate Model")
    
    with st.expander("ℹ️ Theoretical Background: Gaussian Process Regression (GPR)"):
        st.markdown(r"""
        **Gaussian Process Regression (GPR)** is a powerful non-parametric method used to approximate complex functions.
        
        - **Why GPR?**: It provides not just a prediction ($\mu$), but also an uncertainty estimate ($\sigma$) for that prediction.
        - **Kernel Trick**: We use a kernel function $k(x_i, x_j)$ to define the similarity between data points.
            - **Matern Kernel**: General purpose, allows for non-smooth functions.
            - **RBF (Radial Basis Function)**: Smooth functions.
            - **Rational Quadratic**: Infinite sum of RBFs with different length scales.
        - **Auto-Selection**: This dashboard automatically tests multiple kernels using Cross-Validation ($R^2$ score) and selects the best one.
        """)
    
    # Upload or select training data
    data_source = st.radio("Data Source", ["Select from 'data/' folder", "Upload File"])
    
    train_file_path = None
    if data_source == "Select from 'data/' folder":
        # Extensions for training data
        files = []
        for d, label in [(DATA_DIR, 'data/'), (OUTPUT_DIR, 'output/')]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith(('.csv', '.xlsx')):
                        files.append(os.path.join(label, f))
        
        # Try to find default total_2D_Data.xlsx
        default_idx = 0
        if "data/total_2D_Data.xlsx" in files:
            default_idx = files.index("data/total_2D_Data.xlsx")
        
        sel = st.selectbox("Select Training Data", files, index=default_idx if files else None)
        if sel:
             if sel.startswith('data/'):
                train_file_path = os.path.join(DATA_DIR, sel.replace('data/', ''))
             else:
                train_file_path = os.path.join(OUTPUT_DIR, sel.replace('output/', ''))
    else:
        uploaded = st.file_uploader("Upload Excel/CSV", type=['csv', 'xlsx'])
        if uploaded:
            train_file_path = os.path.join(DATA_DIR, uploaded.name)
            with open(train_file_path, "wb") as f:
                f.write(uploaded.getbuffer())
    
    if st.button("Train Model"):
        if train_file_path:
            with st.spinner("Training GP Surrogate Models..."):
                try:
                    models, scaler_X, scalers_y, col_map, abs_path, cv_scores = train_surrogates(train_file_path)
                    
                    # Save model
                    bundle_path = os.path.join(MODEL_DIR, "gp_surrogate_bundle.joblib")
                    meta = {
                        "source": abs_path,
                        "cv_scores": cv_scores
                    }
                    save_surrogate_bundle(models, scaler_X, scalers_y, bundle_path, meta)
                    
                    st.success("Training Complete!")
                    st.json(cv_scores)
                    st.info(f"Model saved to {bundle_path}")
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")
        else:
            st.warning("Please select a training file.")

elif mode == "5. Optimization (GA)":
    st.header("5. Genetic Algorithm Optimization")
    
    with st.expander("ℹ️ Theoretical Background: Genetic Algorithm (GA)"):
        st.markdown(r"""
        **Genetic Algorithm (GA)** is an optimization technique inspired by natural selection.
        
        - **Population**: A set of candidate designs.
        - **Selection**: "Fitter" individuals (better performance) are selected to reproduce.
        - **Crossover**: Combining parameters from two parents to create offspring.
        - **Mutation**: Randomly changing parameters to maintain diversity and avoid local optima.
        
        **Fitness Function (Conservative Approach)**:
        We define the "Fitness" to maximize as the ratio of Heat Flux to Pressure Drop. To be safe, we penalize uncertainty:
        
        $$ F(x) = \frac{\mu_Q - k\sigma_Q}{\mu_{\Delta P} + k\sigma_{\Delta P}} $$
        
        - **$\mu$**: Mean prediction from Surrogate Model.
        - **$\sigma$**: Uncertainty (Standard Deviation) from Surrogate Model.
        - **$k$ (Safety Factor)**: How conservative the design should be. Higher $k$ means we want to be more sure about the performance (lower risk).
        """)
    
    defaults = st.session_state.get('scenario_params', {})
    
    bundle_path = os.path.join(MODEL_DIR, "gp_surrogate_bundle.joblib")
    
    if not os.path.exists(bundle_path):
        st.error("No trained model found! Please go to step 4 and train a model first.")
    else:
        st.write(f"Using model: `{bundle_path}`")
        
        col1, col2 = st.columns(2)
        with col1:
            k_sigma = st.slider("Safety Factor (k_sigma)", 0.0, 3.0, float(defaults.get('k_sigma', 1.0)), 0.1, help="Higher value = More conservative (safer)")
        with col2:
            use_conservative = st.checkbox("Use Conservative Fitness", value=True)
            
        if st.button("Run Optimization"):
            with st.spinner("Running GA Optimization..."):
                models, scaler_X, scalers_y, meta = load_surrogate_bundle(bundle_path)
                result = run_ga(
                    models=models,
                    scaler_X=scaler_X,
                    scalers_y=scalers_y,
                    use_conservative=use_conservative,
                    k_sigma=k_sigma,
                    dp_floor=1e-3,
                    generations=50, # Reduced for dashboard responsiveness
                    pop_size=30
                )
                
                st.success("Optimization Found!")
                
                # Display results
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("Optimal S1", f"{result['S1_mm']:.4f} mm")
                    st.metric("Optimal Fin Height", f"{result['fin_height_fh_mm']:.4f} mm")
                    st.metric("Optimal Fin Spacing", f"{result['fin_spacing_fs_mm']:.4f} mm")
                
                
                with res_col2:
                    st.metric("Predicted Q''", f"{result['Q_mu']:.2f} W/m²")
                    st.metric("Predicted ΔP", f"{result['dP_mu']:.4f} Pa")
                    st.metric("Objective Ratio", f"{result['conservative_ratio' if use_conservative else 'mean_ratio']:.4f}")

                st.info(f"FH Limit for this S1: {result['fh_limit_mm']:.4f} mm")


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if not get_script_run_ctx():
            print("\n\n" + "="*80)
            print("🛑  경고: 이 스크립트는 'python' 명령어로 실행하면 안 됩니다.")
            print(f"    대시보드를 실행하려면 다음 명령어를 사용하세요:")
            print(f"\n    streamlit run {os.path.basename(__file__)}")
            print("\n" + "="*80 + "\n\n")
    except ImportError:
        pass
