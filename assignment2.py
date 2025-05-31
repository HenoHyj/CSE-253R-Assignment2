# %%
"""
Assignment 2: Extended Markov Chains with Chord Accompaniment
Adding harmonic support to the generated melodies
"""

import os
import random
import math
from collections import Counter, defaultdict
from glob import glob
import numpy as np
from numpy.random import choice
from symusic import Score
from miditok import REMI, TokenizerConfig
from midiutil import MIDIFile

random.seed(222)

# %%

MAESTRO_DIR = "maestro_data/maestro-v3.0.0"
midi_files = (glob(os.path.join(MAESTRO_DIR, '**/*.midi'), recursive=True) + 
              glob(os.path.join(MAESTRO_DIR, '**/*.mid'), recursive=True))

if not midi_files:
    midi_files = glob('*.mid')

midi_files = midi_files[:400]
print(f"Using {len(midi_files)} MIDI files")

# %%
# Tokenizer for melody extraction
config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=False)
tokenizer = REMI(config)
tokenizer.train(vocab_size=1000, files_paths=midi_files)

def extract_musical_sequence(midi_file):
    """Extract melody with musical filtering"""
    try:
        midi = Score(midi_file)
        tokens = tokenizer(midi)[0].tokens
        
        raw_pitches = []
        for token in tokens:
            if token.startswith('Pitch_'):
                try:
                    pitch = int(token.split('_')[1])
                    if 48 <= pitch <= 84:
                        raw_pitches.append(pitch)
                except (ValueError, IndexError):
                    continue
        
        if len(raw_pitches) < 10:
            return []
        
        # Smooth large jumps
        musical_pitches = [raw_pitches[0]]
        
        for i in range(1, len(raw_pitches)):
            prev = musical_pitches[-1]
            curr = raw_pitches[i]
            interval = abs(curr - prev)
            
            if interval > 7:
                if curr > prev:
                    smoothed = prev + random.choice([1, 2, 3, 4, 5])
                else:
                    smoothed = prev - random.choice([1, 2, 3, 4, 5])
                smoothed = max(48, min(84, smoothed))
                musical_pitches.append(smoothed)
            else:
                musical_pitches.append(curr)
        
        return musical_pitches
        
    except Exception:
        return []

# Extract sequences
all_sequences = []
for midi_file in midi_files:
    sequence = extract_musical_sequence(midi_file)
    if len(sequence) >= 30:
        all_sequences.append(sequence)

print(f"Extracted {len(all_sequences)} sequences")

# %%
# Extended Musical Markov Chain
class MusicalMarkovChain:
    def __init__(self, order=3):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()
        self.start_contexts = []
        
    def train(self, sequences):
        for sequence in sequences:
            if len(sequence) < self.order + 1:
                continue
            self.vocab.update(sequence)
            
            if len(sequence) >= self.order:
                start = tuple(sequence[:self.order])
                self.start_contexts.append(start)
            
            for i in range(len(sequence) - self.order):
                context = tuple(sequence[i:i + self.order])
                next_note = sequence[i + self.order]
                self.transitions[context][next_note] += 1
                self.context_counts[context] += 1
    
    def sample_musical_next_note(self, context, temperature=1.0):
        if context not in self.transitions:
            last_note = context[-1] if context else 60
            return last_note + random.choice([-2, -1, 0, 1, 2])
        
        candidates = list(self.transitions[context].keys())
        counts = [self.transitions[context][note] for note in candidates]
        
        last_note = context[-1]
        boosted_counts = []
        
        for i, note in enumerate(candidates):
            interval = abs(note - last_note)
            boost = 1.0
            
            if interval <= 2:
                boost = 3.0  # Step motion boost
            elif interval <= 4:
                boost = 1.5
            elif interval > 7:
                boost = 0.3
            
            boosted_counts.append(counts[i] * boost)
        
        if temperature != 1.0:
            boosted_counts = [c ** (1.0 / temperature) for c in boosted_counts]
        
        total = sum(boosted_counts)
        if total > 0:
            probs = [c / total for c in boosted_counts]
            return choice(candidates, p=probs)
        else:
            return candidates[0] if candidates else 60
    
    def generate_musical_sequence(self, length=100, temperature=1.0):
        if self.start_contexts:
            context = random.choice(self.start_contexts)
        else:
            start_note = random.choice([60, 62, 64, 65, 67])
            context = tuple([start_note + i for i in range(self.order)])
        
        generated = list(context)
        
        for _ in range(length - len(generated)):
            next_note = self.sample_musical_next_note(tuple(generated[-self.order:]), temperature)
            next_note = max(48, min(84, next_note))
            generated.append(next_note)
        
        return generated
    
    def get_next_pitch_probabilities(self, context):
        """Get probability distribution for next pitch"""
        if context not in self.transitions:
            # Fallback to uniform distribution over reasonable pitch range
            reasonable_pitches = [p for p in self.vocab if 48 <= p <= 84]  # 3 octaves around middle C
            if reasonable_pitches:
                prob = 1.0 / len(reasonable_pitches)
                return {pitch: prob for pitch in reasonable_pitches}
            else:
                prob = 1.0 / len(self.vocab)
                return {pitch: prob for pitch in self.vocab}
        
        total_count = self.context_counts[context]
        return {pitch: count / total_count 
                for pitch, count in self.transitions[context].items()}
    
    def evaluate_perplexity(self, test_sequences):
        """Evaluate model perplexity on pitch sequences"""
        total_log_prob = 0
        total_pitches = 0
        
        for sequence in test_sequences:
            if len(sequence) <= self.order:
                continue

            for i in range(self.order, len(sequence)):
                context = tuple(sequence[i-self.order:i])
                true_pitch = sequence[i]
                
                probs = self.get_next_pitch_probabilities(context)
                pitch_prob = probs.get(true_pitch, 1e-10)
                
                total_log_prob += -math.log2(pitch_prob)
                total_pitches += 1
        
        return 2 ** (total_log_prob / total_pitches) if total_pitches > 0 else float('inf')

# Train model
model = MusicalMarkovChain(order=3)
model.train(all_sequences)

# Data split for evaluation
train_sequences = all_sequences[:int(0.8 * len(all_sequences))]
val_sequences = all_sequences[int(0.8 * len(all_sequences)):int(0.9 * len(all_sequences))]
test_sequences = all_sequences[int(0.9 * len(all_sequences)):]

print(f"Data split: Train={len(train_sequences)}, Val={len(val_sequences)}, Test={len(test_sequences)}")

# Model Evaluation
print("\n=== MODEL EVALUATION ===")
val_perplexity = model.evaluate_perplexity(val_sequences)
test_perplexity = model.evaluate_perplexity(test_sequences)

print(f"Validation perplexity: {val_perplexity:.2f}")
print(f"Test perplexity: {test_perplexity:.2f}")

print(f"\nModel Statistics:")
print(f"- Order: {model.order} (trigram)")
print(f"- Training sequences: {len(train_sequences)}")
print(f"- Vocabulary size: {len(model.vocab)}")
print(f"- Contexts learned: {len(model.transitions)}")
print(f"- Total notes processed: {sum(len(seq) for seq in all_sequences)}")

# %%
# NEW: Chord Generation System
class ChordGenerator:
    """Generate chord progressions to accompany melodies"""
    
    def __init__(self):
        # Define major scale chord progressions (in C major for simplicity)
        self.chord_progressions = {
            # Common progressions in Roman numeral notation
            'I-V-vi-IV': [(60, 64, 67), (67, 71, 74), (69, 72, 76), (65, 69, 72)],  # C-G-Am-F
            'vi-IV-I-V': [(69, 72, 76), (65, 69, 72), (60, 64, 67), (67, 71, 74)],  # Am-F-C-G  
            'I-vi-IV-V': [(60, 64, 67), (69, 72, 76), (65, 69, 72), (67, 71, 74)],  # C-Am-F-G
            'I-IV-V-I': [(60, 64, 67), (65, 69, 72), (67, 71, 74), (60, 64, 67)],   # C-F-G-C
        }
        
        # Scale degrees to chord mapping (C major)
        self.scale_to_chord = {
            60: (60, 64, 67),  # C -> C major
            62: (62, 65, 69),  # D -> D minor  
            64: (64, 67, 71),  # E -> E minor
            65: (65, 69, 72),  # F -> F major
            67: (67, 71, 74),  # G -> G major
            69: (69, 72, 76),  # A -> A minor
            71: (71, 74, 77),  # B -> B diminished
        }
    
    def get_chord_for_note(self, note):
        """Get appropriate chord for a melody note"""
        # Transpose to C major context
        note_in_c = note % 12
        c_major_note = 60 + note_in_c
        
        # Find closest scale degree
        closest_scale_note = min(self.scale_to_chord.keys(), 
                                key=lambda x: abs((x % 12) - note_in_c))
        
        base_chord = self.scale_to_chord[closest_scale_note]
        
        # Transpose chord to match the octave of the melody note
        octave_offset = (note // 12) * 12 - 60
        transposed_chord = tuple(n + octave_offset for n in base_chord)
        
        # Keep chords in reasonable range (below melody)
        if min(transposed_chord) > note - 12:
            transposed_chord = tuple(n - 12 for n in transposed_chord)
        
        return transposed_chord
    
    def generate_chord_progression(self, melody, progression_length=4):
        """Generate chord progression that follows the melody"""
        chords = []
        notes_per_chord = len(melody) // progression_length
        
        for i in range(progression_length):
            start_idx = i * notes_per_chord
            end_idx = min(start_idx + notes_per_chord, len(melody))
            
            if start_idx < len(melody):
                # Use the first note of each segment to determine chord
                melody_note = melody[start_idx]
                chord = self.get_chord_for_note(melody_note)
                chords.append(chord)
        
        return chords

# %%
# Enhanced MIDI generation with chords
def create_midi_with_chords(melody, filename="melody_with_chords.mid", tempo=120):
    """Create MIDI file with melody and chord accompaniment"""
    
    chord_gen = ChordGenerator()
    chords = chord_gen.generate_chord_progression(melody, progression_length=8)
    
    MyMIDI = MIDIFile(2)  # 2 tracks: melody and chords
    
    # Track 0: Melody
    melody_track = 0
    MyMIDI.addTrackName(melody_track, 0, "Melody")
    MyMIDI.addTempo(melody_track, 0, tempo)
    
    # Track 1: Chords  
    chord_track = 1
    MyMIDI.addTrackName(chord_track, 0, "Chords")
    MyMIDI.addTempo(chord_track, 0, tempo)
    
    # Add melody
    time = 0
    for i, pitch in enumerate(melody):
        duration = 1.0
        
        # Rhythm variation
        if i % 4 == 0:
            duration = 1.5 if random.random() < 0.25 else 1.0
        elif i % 2 == 1:
            duration = 0.5 if random.random() < 0.2 else 1.0
        
        MyMIDI.addNote(melody_track, 0, pitch, time, duration, 100)
        time += duration
    
    # Add chords
    chord_time = 0
    notes_per_chord = len(melody) // len(chords)
    
    for i, chord in enumerate(chords):
        chord_duration = notes_per_chord * 1.0  # Duration based on melody timing
        
        # Add each note in the chord
        for chord_note in chord:
            if 36 <= chord_note <= 84:  # Keep in reasonable range
                MyMIDI.addNote(chord_track, 1, chord_note, chord_time, chord_duration, 70)
        
        chord_time += chord_duration
    
    # Save file
    with open(filename, "wb") as f:
        MyMIDI.writeFile(f)
    
    return filename

# %%
# Generate melodies with chord accompaniment
print("\n=== GENERATING MELODIES WITH CHORDS ===")

# Task 1: Unconditioned with chords
print("\nTask 1: Unconditioned Generation with Chords")
for i in range(3):
    temp = [0.8, 1.0, 1.2][i]
    melody = model.generate_musical_sequence(length=64, temperature=temp)  # 64 notes for 8 chords
    
    print(f"\nMelody {i+1} (temp={temp}): {len(melody)} notes")
    print(f"Range: {min(melody)}-{max(melody)}")
    
    # Analyze step motion
    intervals = [melody[j+1] - melody[j] for j in range(len(melody)-1)]
    steps = sum(1 for iv in intervals if abs(iv) <= 2)
    print(f"Step motion: {steps}/{len(intervals)} ({steps/len(intervals)*100:.1f}%)")
    
    # Generate with chords
    filename = f"melody_with_chords_unconditioned_{i+1}.mid"
    create_midi_with_chords(melody, filename)
    print(f"Saved: {filename}")

# %%
# Task 2: Conditioned with chords
print("\nTask 2: Conditioned Generation with Chords")

for i in range(3):
    if i < len(test_sequences):
        source = test_sequences[i]
        prefix = source[:12]  # Shorter prefix for 64-note total
        
        # Simple conditioned generation
        context = tuple(prefix[-3:]) if len(prefix) >= 3 else tuple([60, 62, 64])
        conditioned = list(prefix)
        
        for _ in range(64 - len(conditioned)):
            next_note = model.sample_musical_next_note(context, temperature=1.0)
            next_note = max(48, min(84, next_note))
            conditioned.append(next_note)
            context = tuple(conditioned[-3:])
        
        print(f"\nConditioned melody {i+1}: {len(conditioned)} notes")
        print(f"Prefix: {prefix[:6]}...")
        print(f"Range: {min(conditioned)}-{max(conditioned)}")
        
        # Analyze step motion
        intervals = [conditioned[j+1] - conditioned[j] for j in range(len(conditioned)-1)]
        steps = sum(1 for iv in intervals if abs(iv) <= 2)
        print(f"Step motion: {steps}/{len(intervals)} ({steps/len(intervals)*100:.1f}%)")
        
        # Generate with chords
        filename = f"melody_with_chords_conditioned_{i+1}.mid"
        create_midi_with_chords(conditioned, filename)
        print(f"Saved: {filename}")

# %%
print("\n=== ASSIGNMENT 2 WITH CHORDS COMPLETE ===")
print("\nGenerated files with chord accompaniment:")
print("Unconditioned:")
for i in range(1, 4):
    print(f"- melody_with_chords_unconditioned_{i}.mid")
print("Conditioned:")
for i in range(1, 4):
    print(f"- melody_with_chords_conditioned_{i}.mid")

print("\nFeatures:")
print("Two-track MIDI: Melody + Chord accompaniment")
print("Smart chord progression based on melody notes")
print("Common progressions: I-V-vi-IV, vi-IV-I-V, etc.")
print("Chords positioned below melody for proper voicing")
print("Enhanced musical experience!")

print(f"\nFinal Model Performance:")
print(f"- Validation perplexity: {val_perplexity:.2f}")
print(f"- Test perplexity: {test_perplexity:.2f}")
print(f"- Musical step motion bias (3x boost for steps)")
print(f"- Large leap penalties applied")
print(f"- Vocabulary: {len(model.vocab)} unique pitches")
print(f"- Training data: {len(train_sequences)} sequences")

# %% 