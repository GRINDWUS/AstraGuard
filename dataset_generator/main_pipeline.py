"""
Main Dataset Generation & Partitioning Pipeline
================================================
Generates 5 independent qualification lots, partitions them into train/val/test splits,
and persists them into validation/dataset/ alongside metadata.json.
"""

import os
import json
import numpy as np
import pandas as pd
from dataset_generator.lot_generator import LotSimulator

def generate_complete_dataset(output_dir: str = 'validation/dataset'):
    os.makedirs(output_dir, exist_ok=True)
    all_lots = []
    
    for lot_id in range(5):
        print(f"Generating Lot {lot_id:02d}...")
        base_iddq = 1.2 + np.random.uniform(-0.1, 0.1)
        ea = 0.68 + np.random.uniform(-0.02, 0.02)
        
        simulator = LotSimulator(base_iddq=base_iddq, ea_eV=ea)
        lot_df = simulator.generate_lot(lot_id=lot_id, num_components=2000)
        all_lots.append(lot_df)
        
        lot_path = os.path.join(output_dir, f'lot_{lot_id:02d}.csv')
        lot_df.to_csv(lot_path, index=False)
        print(f"  ✅ Saved {lot_path}")

    # Concatenate train (Lots 0-3) and test (Lot 4)
    train_data = pd.concat(all_lots[0:4], ignore_index=True)
    train_path = os.path.join(output_dir, 'train_lot_0_3.csv')
    train_data.to_csv(train_path, index=False)
    print(f"✅ Train set saved to {train_path} ({len(train_data)} components)")

    val_data = all_lots[2].sample(frac=0.5, random_state=42)
    val_path = os.path.join(output_dir, 'val_lot_2_subset.csv')
    val_data.to_csv(val_path, index=False)
    print(f"✅ Val set saved to {val_path} ({len(val_data)} components)")

    test_data = all_lots[4]
    test_path = os.path.join(output_dir, 'test_lot_4.csv')
    test_data.to_csv(test_path, index=False)
    print(f"✅ Test set saved to {test_path} ({len(test_data)} components)")

    full_df = pd.concat(all_lots, ignore_index=True)
    metadata = {
        'total_components': len(full_df),
        'train_components': len(train_data),
        'val_components': len(val_data),
        'test_components': len(test_data),
        'nominal_rate': float((full_df['failure_mode_gt'] == 'NOMINAL').mean()),
        'failure_rate': float((full_df['failure_mode_gt'] != 'NOMINAL').mean()),
        'activation_energy_baseline_eV': 0.68,
        'standards': ['MIL-STD-883 Method 1015', 'ESCC 9000 Level B/S']
    }

    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Dataset Pipeline Complete!")
    print(f"Total Components: {metadata['total_components']}")
    print(f"Nominal Yield: {metadata['nominal_rate']*100:.2f}% | Failure Yield: {metadata['failure_rate']*100:.2f}%")
    return full_df, metadata

if __name__ == '__main__':
    generate_complete_dataset()
