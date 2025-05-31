# %%
"""
Assignment 2: Extended Markov Chains with LEARNED Chord Progressions
Both melody and chords generated from Extended Markov Chain models
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
# Configuration
MAESTRO_DIR = "maestro_data/maestro-v3.0.0"
midi_files = (glob(os.path.join(MAESTRO_DIR, '**/*.midi'), recursive=True) + 
              glob(os.path.join(MAESTRO_DIR, '**/*.mid'), recursive=True))

if not midi_files:
    midi_files = glob('*.mid')

midi_files = midi_files[:200]  # Use more files for better chord learning
print(f"Using {len(midi_files)} MIDI files")

# %%
# Enhanced tokenizer and data extraction
config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=False)
tokenizer = REMI(config)
tokenizer.train(vocab_size=1000, files_paths=midi_files)

def extract_melody_and_chord_data(midi_file):
    """Extract both melody and chord information from MIDI"""
    try:
        midi = Score(midi_file)
        tokens = tokenizer(midi)[0].tokens
        
        # Extract all pitches with their positions
        pitch_events = []
        current_position = 0
        
        for token in tokens:
            if token.startswith('Position_'):
                try:
                    current_position = int(token.split('_')[1])
                except:
                    continue
            elif token.startswith('Pitch_'):
                try:
                    pitch = int(token.split('_')[1])
                    if 36 <= pitch <= 96:
                        pitch_events.append((current_position, pitch))
                except:
                    continue
        
        if len(pitch_events) < 20:
            return [], []
        
        # Group pitches by position to identify chords
        position_groups = defaultdict(list)
        for pos, pitch in pitch_events:
            position_groups[pos].append(pitch)
        
        # Extract melody and chords with better pitch selection
        melody = []
        chord_sequence = []
        
        sorted_positions = sorted(position_groups.keys())
        
        for pos in sorted_positions:
            pitches = sorted(position_groups[pos])
            
            # IMPROVED: Choose melody note more intelligently
            if len(pitches) == 1:
                melody_note = pitches[0]
            elif len(pitches) >= 4:
                # For dense chords, choose second highest (not always highest)
                melody_note = pitches[-2]
            else:
                # For 2-3 notes, choose highest
                melody_note = pitches[-1]
            
            melody.append(melody_note)
            
            # Chord: if multiple pitches, extract chord root
            if len(pitches) >= 2:
                # Use lowest pitch as chord root, then build chord
                root = pitches[0]
                chord_type = classify_chord(pitches)
                chord_sequence.append((root % 12, chord_type))  # Normalize to pitch class
            else:
                # Single note - infer chord from melody
                inferred_chord = infer_chord_from_melody(melody_note)
                chord_sequence.append(inferred_chord)
        
        return melody, chord_sequence
        
    except Exception as e:
        return [], []

def classify_chord(pitches):
    """Classify chord type based on interval pattern"""
    if len(pitches) < 2:
        return 'single'
    
    # Normalize to pitch classes
    pc_set = sorted(set(p % 12 for p in pitches))
    
    if len(pc_set) < 2:
        return 'single'
    elif len(pc_set) == 2:
        return 'dyad'
    else:
        # Check for common triads
        intervals = [(pc_set[i+1] - pc_set[i]) % 12 for i in range(len(pc_set)-1)]
        
        if 3 in intervals and 4 in intervals:  # Major or minor triad
            if intervals[:2] == [4, 3]:
                return 'major'
            elif intervals[:2] == [3, 4]:
                return 'minor'
            else:
                return 'other'
        else:
            return 'other'

def infer_chord_from_melody(melody_note):
    """Infer likely chord from melody note using music theory"""
    # Map melody notes to likely chord roots (in C major context)
    root_mapping = {
        0: (0, 'major'),    # C -> C major
        1: (5, 'major'),    # C# -> F major (common substitution)
        2: (2, 'minor'),    # D -> D minor
        3: (2, 'minor'),    # D# -> D minor
        4: (4, 'minor'),    # E -> E minor
        5: (5, 'major'),    # F -> F major
        6: (5, 'major'),    # F# -> F major
        7: (7, 'major'),    # G -> G major
        8: (7, 'major'),    # G# -> G major
        9: (9, 'minor'),    # A -> A minor
        10: (9, 'minor'),   # A# -> A minor
        11: (11, 'dim'),    # B -> B diminished
    }
    
    melody_pc = melody_note % 12
    return root_mapping.get(melody_pc, (0, 'major'))

def smooth_melody(pitches):
    if len(pitches) <= 1:
        return pitches
    
    smoothed = [pitches[0]]
    for i in range(1, len(pitches)):
        prev = smoothed[-1]
        curr = pitches[i]
        
        if abs(curr - prev) > 7:
            if curr > prev:
                smoothed_note = prev + random.choice([1, 2, 3, 4, 5])
            else:
                smoothed_note = prev - random.choice([1, 2, 3, 4, 5])
            smoothed_note = max(48, min(84, smoothed_note))
            smoothed.append(smoothed_note)
        else:
            smoothed.append(curr)
    
    return smoothed

def normalize_pitch_range(melody, target_min=60, target_max=84):
    """Normalize melody to a comfortable pitch range (C4-C6)"""
    if not melody:
        return melody
    
    current_min = min(melody)
    current_max = max(melody)
    current_range = current_max - current_min
    
    # If the melody is too high, transpose down
    if current_min > 72:  # If lowest note is above C5
        transpose = -12  # Transpose down an octave
    elif current_min > 84:  # If lowest note is above C6
        transpose = -24  # Transpose down two octaves
    # If the melody is too low, transpose up
    elif current_max < 48:  # If highest note is below C3
        transpose = 12   # Transpose up an octave
    elif current_max < 36:  # If highest note is below C2
        transpose = 24   # Transpose up two octaves
    else:
        transpose = 0
    
    # Apply transposition
    normalized = [max(48, min(84, pitch + transpose)) for pitch in melody]
    
    return normalized

# %%
# Extract melody and chord data
print("Extracting melody and chord sequences...")
melody_sequences = []
chord_sequences = []

for midi_file in midi_files:
    melody, chords = extract_melody_and_chord_data(midi_file)
    if len(melody) >= 30 and len(chords) >= 30:
        # Apply melody smoothing
        smoothed_melody = smooth_melody(melody)
        # Apply pitch range normalization
        normalized_melody = normalize_pitch_range(smoothed_melody)
        melody_sequences.append(normalized_melody)
        chord_sequences.append(chords)

print(f"Extracted {len(melody_sequences)} melody sequences")
print(f"Extracted {len(chord_sequences)} chord sequences")

# %%
# Musical Markov Chain for melodies
class MusicalMarkovChain:
    def __init__(self, order=3):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()
        self.start_contexts = []
        
    def train(self, sequences):
        print(f"Training melody {self.order}-gram model...")
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
        
        print(f"Melody model: {len(self.transitions)} contexts, {len(self.vocab)} pitches")
    
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
                boost = 3.0
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
            # Use comfortable mid-range starting notes (C4-G4)
            start_note = random.choice([60, 62, 64, 65, 67])  # C4, D4, E4, F4, G4
            context = tuple([start_note + i for i in range(self.order)])
        
        generated = list(context)
        
        for _ in range(length - len(generated)):
            next_note = self.sample_musical_next_note(tuple(generated[-self.order:]), temperature)
            next_note = max(48, min(84, next_note))  # Keep in C3-C6 range
            generated.append(next_note)
        
        return generated

# %%
# NEW: Extended Markov Chain for Chord Progressions
class ChordMarkovChain:
    """Extended Markov Chain that LEARNS chord progressions from data"""
    
    def __init__(self, order=2):  # Bigram for chords
        self.order = order
        self.transitions = defaultdict(Counter)
        self.context_counts = Counter()
        self.chord_vocab = set()
        self.start_contexts = []
        
    def train(self, chord_sequences):
        print(f"Training chord {self.order}-gram model...")
        
        for sequence in chord_sequences:
            if len(sequence) < self.order + 1:
                continue
                
            self.chord_vocab.update(sequence)
            
            # Store start contexts
            if len(sequence) >= self.order:
                start = tuple(sequence[:self.order])
                self.start_contexts.append(start)
            
            # Build transitions
            for i in range(len(sequence) - self.order):
                context = tuple(sequence[i:i + self.order])
                next_chord = sequence[i + self.order]
                
                self.transitions[context][next_chord] += 1
                self.context_counts[context] += 1
        
        print(f"Chord model: {len(self.transitions)} contexts, {len(self.chord_vocab)} unique chords")
    
    def sample_next_chord(self, context, temperature=1.0):
        """Sample next chord given context"""
        if context not in self.transitions:
            # Fallback to common chords
            common_chords = [(0, 'major'), (7, 'major'), (9, 'minor'), (5, 'major')]
            available_chords = [c for c in common_chords if c in self.chord_vocab]
            if available_chords:
                return random.choice(available_chords)
            else:
                return (0, 'major')
        
        candidates = list(self.transitions[context].keys())
        probs = [self.transitions[context][chord] / self.context_counts[context] 
                for chord in candidates]
        
        # Apply temperature
        if temperature != 1.0:
            probs = [p ** (1.0 / temperature) for p in probs]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
        
        if len(candidates) > 0 and len(probs) > 0:
            return random.choices(candidates, weights=probs)[0]
        else:
            return (0, 'major')
    
    def generate_chord_progression(self, length=8, temperature=1.0):
        """Generate a learned chord progression"""
        if not self.start_contexts:
            # Default progression if no training data
            return [(0, 'major'), (7, 'major'), (9, 'minor'), (5, 'major')] * (length // 4)
        
        context = random.choice(self.start_contexts)
        generated = list(context)
        
        for _ in range(length - len(generated)):
            next_chord = self.sample_next_chord(context, temperature)
            generated.append(next_chord)
            context = tuple(generated[-self.order:])
        
        return generated[:length]

# %%
# Train both models
melody_model = MusicalMarkovChain(order=3)
melody_model.train(melody_sequences)

chord_model = ChordMarkovChain(order=2)  # Bigram for chords
chord_model.train(chord_sequences)

# %%
# Enhanced MIDI generation with LEARNED chords
def create_midi_with_learned_chords(melody, filename="melody_with_learned_chords.mid", tempo=120):
    """Create MIDI file with melody and LEARNED chord accompaniment"""
    
    # Generate chord progression using the trained model
    chord_progression_length = len(melody) // 8  # One chord per 8 melody notes
    learned_chords = chord_model.generate_chord_progression(chord_progression_length)
    
    MyMIDI = MIDIFile(2)  # 2 tracks: melody and chords
    
    # Track 0: Melody
    melody_track = 0
    MyMIDI.addTrackName(melody_track, 0, "Melody")
    MyMIDI.addTempo(melody_track, 0, tempo)
    
    # Track 1: Learned Chords  
    chord_track = 1
    MyMIDI.addTrackName(chord_track, 0, "Learned Chords")
    MyMIDI.addTempo(chord_track, 0, tempo)
    
    # Add melody
    time = 0
    for i, pitch in enumerate(melody):
        duration = 1.0
        
        if i % 4 == 0:
            duration = 1.5 if random.random() < 0.25 else 1.0
        elif i % 2 == 1:
            duration = 0.5 if random.random() < 0.2 else 1.0
        
        MyMIDI.addNote(melody_track, 0, pitch, time, duration, 100)
        time += duration
    
    # Add learned chords
    chord_time = 0
    notes_per_chord = len(melody) // len(learned_chords)
    
    for i, (root, chord_type) in enumerate(learned_chords):
        chord_duration = notes_per_chord * 1.0
        
        # Convert learned chord to actual pitches
        chord_pitches = chord_symbol_to_pitches(root, chord_type)
        
        # Add each note in the chord
        for chord_note in chord_pitches:
            if 36 <= chord_note <= 84:
                MyMIDI.addNote(chord_track, 1, chord_note, chord_time, chord_duration, 70)
        
        chord_time += chord_duration
    
    with open(filename, "wb") as f:
        MyMIDI.writeFile(f)
    
    return filename

def chord_symbol_to_pitches(root, chord_type):
    """Convert chord symbol to actual MIDI pitches"""
    # Base octave for chords (below middle C)
    base_octave = 48  # C3
    root_pitch = base_octave + root
    
    if chord_type == 'major':
        return [root_pitch, root_pitch + 4, root_pitch + 7]  # Root, major third, fifth
    elif chord_type == 'minor':
        return [root_pitch, root_pitch + 3, root_pitch + 7]  # Root, minor third, fifth
    elif chord_type == 'dim':
        return [root_pitch, root_pitch + 3, root_pitch + 6]  # Root, minor third, dim fifth
    elif chord_type == 'dyad':
        return [root_pitch, root_pitch + 7]  # Root and fifth only
    else:  # 'other' or 'single'
        return [root_pitch, root_pitch + 4, root_pitch + 7]  # Default to major

# %%
# Generate melodies with LEARNED chord accompaniment
print("\n=== GENERATING WITH LEARNED CHORDS ===")

# Task 1: Unconditioned with learned chords
print("\nTask 1: Unconditioned Generation with Learned Chords")
for i in range(3):
    temp = [0.8, 1.0, 1.2][i]
    melody = melody_model.generate_musical_sequence(length=64, temperature=temp)
    
    print(f"\nMelody {i+1} (temp={temp}): {len(melody)} notes")
    print(f"Range: {min(melody)}-{max(melody)}")
    
    # Analyze step motion
    intervals = [melody[j+1] - melody[j] for j in range(len(melody)-1)]
    steps = sum(1 for iv in intervals if abs(iv) <= 2)
    print(f"Step motion: {steps}/{len(intervals)} ({steps/len(intervals)*100:.1f}%)")
    
    # Generate with LEARNED chords
    filename = f"learned_chords_unconditioned_{i+1}.mid"
    create_midi_with_learned_chords(melody, filename)
    print(f"Saved: {filename}")

# Task 2: Conditioned with learned chords
print("\nTask 2: Conditioned Generation with Learned Chords")
test_melodies = melody_sequences[int(0.9 * len(melody_sequences)):]

for i in range(3):
    if i < len(test_melodies):
        source = test_melodies[i]
        prefix = source[:12]
        
        # Generate conditioned melody
        context = tuple(prefix[-3:]) if len(prefix) >= 3 else tuple([60, 62, 64])
        conditioned = list(prefix)
        
        for _ in range(64 - len(conditioned)):
            next_note = melody_model.sample_musical_next_note(context, temperature=1.0)
            next_note = max(48, min(84, next_note))
            conditioned.append(next_note)
            context = tuple(conditioned[-3:])
        
        print(f"\nConditioned melody {i+1}: {len(conditioned)} notes")
        print(f"Prefix: {prefix[:6]}...")
        print(f"Range: {min(conditioned)}-{max(conditioned)}")
        
        intervals = [conditioned[j+1] - conditioned[j] for j in range(len(conditioned)-1)]
        steps = sum(1 for iv in intervals if abs(iv) <= 2)
        print(f"Step motion: {steps}/{len(intervals)} ({steps/len(intervals)*100:.1f}%)")
        
        filename = f"learned_chords_conditioned_{i+1}.mid"
        create_midi_with_learned_chords(conditioned, filename)
        print(f"Saved: {filename}")

# %%
print(f"\nModel Statistics:")
print(f"- Melody vocabulary: {len(melody_model.vocab)} unique pitches")
print(f"- Chord vocabulary: {len(chord_model.chord_vocab)} unique chord types")
print(f"- Melody contexts: {len(melody_model.transitions)}")
print(f"- Chord contexts: {len(chord_model.transitions)}")

# %%
# EVALUATION SECTION (matching original assignment2.py)
print("\n=== EVALUATION ===")

# Split data into train/validation/test
train_melodies = melody_sequences[:int(0.8 * len(melody_sequences))]
val_melodies = melody_sequences[int(0.8 * len(melody_sequences)):int(0.9 * len(melody_sequences))]
test_melodies = melody_sequences[int(0.9 * len(melody_sequences)):]

print(f"Data split: {len(train_melodies)} train, {len(val_melodies)} val, {len(test_melodies)} test")

# Calculate perplexity on test set
def calculate_perplexity(model, sequences):
    """Calculate perplexity of the model on given sequences"""
    total_log_prob = 0
    total_tokens = 0
    
    for sequence in sequences:
        if len(sequence) <= model.order:
            continue
            
        for i in range(len(sequence) - model.order):
            context = tuple(sequence[i:i + model.order])
            next_token = sequence[i + model.order]
            
            if context in model.transitions:
                count = model.transitions[context][next_token]
                total_count = model.context_counts[context]
                prob = count / total_count
                
                if prob > 0:
                    total_log_prob += math.log(prob)
                    total_tokens += 1
    
    if total_tokens == 0:
        return float('inf')
    
    avg_log_prob = total_log_prob / total_tokens
    perplexity = math.exp(-avg_log_prob)
    return perplexity

# Calculate perplexity for melody model
melody_perplexity = calculate_perplexity(melody_model, test_melodies)
print(f"Melody model perplexity on test set: {melody_perplexity:.2f}")

# Calculate perplexity for chord model
test_chords = chord_sequences[int(0.9 * len(chord_sequences)):]
chord_perplexity = calculate_perplexity(chord_model, test_chords)
print(f"Chord model perplexity on test set: {chord_perplexity:.2f}")

# Analyze musical quality metrics
print("\n=== MUSICAL QUALITY ANALYSIS ===")

def analyze_musical_quality(melodies, name):
    """Analyze musical quality metrics"""
    all_intervals = []
    all_ranges = []
    step_motions = []
    
    for melody in melodies:
        if len(melody) < 2:
            continue
            
        # Calculate intervals
        intervals = [melody[i+1] - melody[i] for i in range(len(melody)-1)]
        all_intervals.extend(intervals)
        
        # Calculate range
        melody_range = max(melody) - min(melody)
        all_ranges.append(melody_range)
        
        # Calculate step motion percentage
        steps = sum(1 for iv in intervals if abs(iv) <= 2)
        if len(intervals) > 0:
            step_motions.append(steps / len(intervals) * 100)
    
    print(f"\n{name} Analysis:")
    print(f"- Average step motion: {np.mean(step_motions):.1f}%")
    print(f"- Average range: {np.mean(all_ranges):.1f} semitones")
    print(f"- Interval distribution:")
    interval_counts = Counter(all_intervals)
    for interval in sorted(interval_counts.keys())[:10]:  # Show top 10
        count = interval_counts[interval]
        percentage = count / len(all_intervals) * 100
        print(f"  {interval:+2d}: {percentage:5.1f}%")

# Analyze training data
analyze_musical_quality(train_melodies, "Training Data")

# %% 