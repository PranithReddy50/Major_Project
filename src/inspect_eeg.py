import pandas as pd
import sys

try:
    df = pd.read_excel('EEG_Signals_acquiredDataset.xlsx')
    with open('eeg_summary.txt', 'w') as f:
        f.write(str(df.head()) + '\n')
        f.write(str(df.columns) + '\n')
        f.write(f"Shape: {df.shape}\n")
except Exception as e:
    with open('eeg_summary.txt', 'w') as f:
        f.write(f"Error: {e}")
