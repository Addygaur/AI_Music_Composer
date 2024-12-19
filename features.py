import json
import os
import pandas as pd

# Path to the converted folder
folder_path = "data/emopia/_converted/"

class features_extract:
    def traverse_files():
        # Loop through all JSON files and extract features
        all_features = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".json"):
                json_path = os.path.join(folder_path, file_name)
                features = features_extract.extract_features(json_path)
                all_features.append(features)

        # Convert to a DataFrame for analysis
        df = pd.DataFrame(all_features)

        # Save to a CSV for inspection or further processing
        df.to_csv("emopia_features.csv", index=False)
    
    # Function to extract relevant fields from a JSON file
    def extract_features(json_file):
        with open(json_file, "r") as file:
            data = json.load(file)
            
            # Extract metadata
            emo_class = data["annotations"][0]["annotation"]["emo_class"]  # Emotion class
            segment_id = data["annotations"][0]["annotation"]["seg_id"]    # Segment ID
            source_filename = data["metadata"]["source_filename"]
            source_filename = source_filename.split('.mid')[0]
            
            # Extract tempo
            tempo = data["tempos"][0]["qpm"] if data["tempos"] else 120.0  # Default to 120 if missing
            
            # Extract time signature
            if data["time_signatures"]:
                time_sig = data["time_signatures"][0]
                time_signature = f"{time_sig['numerator']}/{time_sig['denominator']}"
            else:
                time_signature = "4/4"  # Default to common time
            
            # Extract track details (notes)
            notes = []
            for track in data["tracks"]:
                if not track["is_drum"]:  # Only include non-drum tracks
                    for note in track["notes"]:
                        notes.append((note["pitch"], note["duration"], note["velocity"], note["time"]))
            
            # Compute aggregated note features
            pitches = [note[0] for note in notes]
            durations = [note[1] for note in notes]
            velocities = [note[2] for note in notes]
            
            features = {
                "source_filename": source_filename,
                "emo_class": emo_class,
                "segment_id": segment_id,
                "tempo": tempo,
                "time_signature": time_signature,
                "mean_pitch": sum(pitches) / len(pitches) if pitches else 0,
                "mean_duration": sum(durations) / len(durations) if durations else 0,
                "mean_velocity": sum(velocities) / len(velocities) if velocities else 0,
                "note_count": len(notes)
            }
            
            return features


