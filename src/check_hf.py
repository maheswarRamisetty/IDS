from datasets import load_dataset
import pandas as pd

ds = load_dataset("yeong-hwan/2024-earnings-call-transcript")

for split in ds.keys():
    df = pd.DataFrame(ds[split])

    # Save to CSV
    csv_filename = f"earnings_call_transcript_{split}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved: {csv_filename}")
