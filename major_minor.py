from music21 import converter, key
import os
import pandas as pd

# Directory containing your MIDI files
midi_dir = "data/emopia/EMOPIA_2.2/midis"
data = []

def extract_major_minor_ratio(midi_file):
    try:
        # Load MIDI file
        midi = converter.parse(midi_file)
        # Analyze key signature
        key_signature = midi.analyze('Krumhansl')
        if key_signature.mode == 'major':
            return 1, 0  # Major key
        elif key_signature.mode == 'minor':
            return 0, 1  # Minor key
    except:
        pass
    return 0, 0  # Default if analysis fails


def process_mid():

    # Process MIDI files
    for midi_file in os.listdir(midi_dir):
        if midi_file.endswith(".mid"):
            file_path = os.path.join(midi_dir, midi_file)
            major_ratio, minor_ratio = extract_major_minor_ratio(file_path)
            filename = midi_file.split('.mid')[0]
            print(filename)
            data.append({"filename": filename, "major_key_ratio": major_ratio, "minor_key_ratio": minor_ratio})

    # Save results to a CSV
    key_data = pd.DataFrame(data)
    key_data.to_csv("key_features.csv", index=False)
