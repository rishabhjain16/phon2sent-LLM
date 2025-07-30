import json
import pandas as pd
import os
import argparse

def create_simple_csv(json_file, output_file=None):
    """
    Create a simple CSV with just Reference, Hypothesis, and Reconstructed columns
    """
    # Read the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Determine output file if not provided
    if output_file is None:
        output_file = os.path.splitext(json_file)[0] + "_simple.csv"
    
    # Extract the relevant fields
    utt_ids = data['utt_id'] if 'utt_id' in data else range(len(data['ref']))
    refs = data['ref']
    hypos = data['hypo']
    reconstructed = data['reconstructed']
    
    # Create a DataFrame
    df = pd.DataFrame({
        'ID': utt_ids,
        'Reference': refs,
        'Hypothesis': hypos,
        'Reconstructed': reconstructed
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} entries to {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Create a simple CSV with just Reference, Hypothesis, and Reconstructed columns")
    parser.add_argument("json_file", help="Path to the JSON results file")
    parser.add_argument("--output", "-o", help="Path to the output CSV file (default: based on JSON file)")
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"Error: File not found: {args.json_file}")
        return
    
    create_simple_csv(args.json_file, args.output)

if __name__ == "__main__":
    main() 