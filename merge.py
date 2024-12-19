import pandas as pd

def merge():

    # Load the CSV files
    df1 = pd.read_csv('emopia_features.csv')  # First CSV with source_filename
    df2 = pd.read_csv('data/emopia/EMOPIA_2.2/key_mode_tempo.csv')  # Second CSV with name and tempo

    # Create a mapping of 'name' to 'tempo' from the second CSV
    tempo_mapping = dict(zip(df2['name'], df2['tempo']))

    # Update the 'tempo' column in the first CSV
    df1['tempo'] = df1['source_filename'].map(tempo_mapping).fillna(df1['tempo'])

    # Save the updated DataFrame if needed
    df1.to_csv('emopia_features.csv', index=False)

    print("Updated DataFrame:")
