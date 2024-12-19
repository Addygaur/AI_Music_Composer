import json
import os

data_path = "data/emopia/_converted"
sequences = []

for file in os.listdir(data_path):
    if file.endswith('json'):
        with open(os.path.join(data_path, file), "r") as f:
            content = json.load(f)
            emo_class = int(content["annotations"][0]["annotation"]["emo_class"])
            for track in content["tracks"]:
                if not track["is_drum"]:
                    notes = track["notes"]
                    sequences.append({"notes": notes, "emo_class": emo_class})

print(sequences)


import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# Extract sequences and normalize
note_features = []
emo_labels = []

for sequence in sequences:
    notes = sequence["notes"]
    emo_class = sequence["emo_class"]
    
    # Extract features for each note
    features = [
        [note["time"], note["pitch"], note["duration"], note["velocity"]]
        for note in notes
    ]
    
    # Convert to numpy and normalize
    features = np.array(features, dtype=np.float32)
    features[:, 0] /= features[:, 0].max()  # Normalize time
    features[:, 1] /= 127.0  # Normalize pitch (MIDI pitch range is 0-127)
    features[:, 2] /= features[:, 2].max()  # Normalize duration
    features[:, 3] /= 127.0  # Normalize velocity

    note_features.append(torch.tensor(features, dtype=torch.float32))
    emo_labels.append(emo_class - 1)  # Convert emo_class to 0-based indexing

# Pad sequences
padded_sequences = pad_sequence(note_features, batch_first=True)
emo_labels = torch.tensor(emo_labels, dtype=torch.long)

print("Padded Sequences Shape:", padded_sequences.shape)
print("Emotion Labels Shape:", emo_labels.shape)


from sklearn.model_selection import train_test_split

# Split data
train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    padded_sequences, emo_labels, test_size=0.2, random_state=42, stratify=emo_labels
)


from torch.utils.data import DataLoader, TensorDataset

# Create DataLoader
train_dataset = TensorDataset(train_sequences, train_labels)
test_dataset = TensorDataset(test_sequences, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


import torch.nn as nn

class MusicGeneratorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MusicGeneratorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM output
        out = self.fc(out)  # Linear layer for prediction
        return out

# Define model
input_size = 4  # Features: time, pitch, duration, velocity
hidden_size = 128
num_layers = 2
output_size = input_size  # Predict next note (same features)

model = MusicGeneratorLSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()  # Regression loss for sequence generation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



num_epochs = 40

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for inputs, labels in train_loader: #For this 0th epoch, only 32 mid sequences: train loader consisting of 32 notes
        # Shift inputs to predict the next note
        inputs, targets = inputs[:, :-1, :], inputs[:, 1:, :]
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

# Save the trained model
model_save_path = "music_generator_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': input_size,
    'hidden_size': hidden_size,
    'output_size': output_size,
}, "music_generator_model.pth")

def generate_music(model, initial_sequence, mood_class, sequence_length=50):
    model.eval()
    generated_sequence = initial_sequence.clone()

    for _ in range(sequence_length):
        with torch.no_grad():
            output = model(generated_sequence.unsqueeze(0))
            next_note = output[:, -1, :]  # Take the last note prediction
            generated_sequence = torch.cat([generated_sequence, next_note], dim=0)
    
    return generated_sequence

# Example usage
initial_sequence = train_sequences[0][:10]  # Use the first 10 notes of a sequence
generated_sequence = generate_music(model, initial_sequence, mood_class=1)
print(generated_sequence)


import pretty_midi

def create_midi_from_sequence(sequence, output_file="generated_music.mid"):
    """
    Converts a sequence of notes into a MIDI file.
    
    Args:
        sequence (torch.Tensor): Generated sequence with columns [time, pitch, duration, velocity].
        output_file (str): Name of the MIDI file to save.
    """
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Default piano instrument
    
    # Add each note to the instrument
    current_time = 0.0
    for note in sequence:
        # Denormalize the features (adjust according to your scaling during preprocessing)
        pitch = int(note[1] * 127)
        start_time = current_time + note[0].item()  # Add note's relative start time
        duration = note[2].item()  # Denormalized duration
        velocity = int(note[3] * 127)
        
        # Create a PrettyMIDI Note
        midi_note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start_time,
            end=start_time + duration
        )
        instrument.notes.append(midi_note)
        current_time = start_time  # Update time to the note's start time
    
    # Add the instrument to the MIDI object
    midi.instruments.append(instrument)
    
    # Write to a MIDI file
    midi.write(output_file)
    print(f"MIDI file saved as {output_file}")


# Convert and save as MIDI
create_midi_from_sequence(generated_sequence)



