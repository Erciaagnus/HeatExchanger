
import os
import sys
import numpy as np
import pandas as pd

# Add scripts to path to reuse logic
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from sampling.generate_point import generate_constrained_lhs
from sampling.mapping_to_porous import run_batch_mapping, calculate_porous_parameters

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_data():
    print("Generating sample data...")
    
    # 1. LHS Samples
    lhs_path = os.path.join(OUTPUT_DIR, 'sample_design_LHS.csv')
    df_lhs = generate_constrained_lhs(n=50, seed=42, out_path=lhs_path)
    print(f"Created {lhs_path}")
    
    # 2. Porous Mapping
    porous_path = os.path.join(OUTPUT_DIR, 'sample_porous_data.csv')
    df_porous = run_batch_mapping(
        in_csv=lhs_path,
        out_csv=porous_path,
        T_celsius=25.0,
        v_design=2.5,
        Dc_mm=24.0,
        delta_f_mm=0.5,
        pitch_ratio=1.0,
        N=4,
        v_min=0.5,
        v_max=3.5,
        n_points=10,
        max_rows=None,
        progress_every=10,
        check_constraint=True
    )
    print(f"Created {porous_path}")
    
    # 3. Synthetic Training Data (Mock CFD)
    # We use the calculated values from mapping but rename columns to match what the Surrogate model expects
    # Surrogate expects: S1, FH, FS, Q'', Delta P
    
    df_train = df_porous.copy()
    
    # Check what columns we have
    # df_porous has: S1_mm, fin_height_fh_mm, fin_spacing_fs_mm
    # and design point outputs: h_fs_W_m2K, dP_total_Pa
    
    # We need heat flux Q'' (W/m^2). 
    # Let's approximate Q'' ~ h_fs * (T_wall - T_bulk).
    # Assume Delta T = 10 K for demonstration.
    df_train["Q'' [W/m^2]"] = df_train["h_fs_W_m2K"] * 10.0 + np.random.normal(0, 50, len(df_train)) # Add noise
    
    # Rename Pressure Drop
    df_train["Delta P [Pa]"] = df_train["dP_total_Pa"] + np.random.normal(0, 10, len(df_train)) # Add noise
    
    # Select relevant columns for training file
    # Columns expected by train_model.py auto-detection:
    # S1, Height, Spacing, Q'', Delta P
    
    training_data = df_train[[
        "S1_mm", 
        "fin_height_fh_mm", 
        "fin_spacing_fs_mm", 
        "Q'' [W/m^2]", 
        "Delta P [Pa]"
    ]].copy()
    
    train_path = os.path.join(OUTPUT_DIR, 'sample_training_data.csv')
    training_data.to_csv(train_path, index=False)
    print(f"Created {train_path}")

if __name__ == "__main__":
    generate_data()
