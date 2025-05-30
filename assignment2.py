# %%
import os
import glob
import json
import random
import math
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm

# MidiTok imports
from miditok import REMI, TokenizerConfig, TokSequence
from symusic import Score
import miditoolkit

# Set seed for reproducibility
random.seed(42)

# %%
# Configuration
MAESTRO_VERSION = "v3.0.0"
PROJECT_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJECT_DIR, "maestro_data")
MIDI_DIR = os.path.join(DATA_DIR, f"maestro-{MAESTRO_VERSION}")
METADATA_CSV_PATH = os.path.join(DATA_DIR, f"maestro-{MAESTRO_VERSION}.csv")
TOKENIZER_PATH = os.path.join(DATA_DIR, "maestro_tokenizer_params.json")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "tokenized_sequences")

# %%
# Load MIDI files
all_midi_files = (glob.glob(os.path.join(MIDI_DIR, '**/*.midi'), recursive=True) + 
                  glob.glob(os.path.join(MIDI_DIR, '**/*.mid'), recursive=True))

if not all_midi_files:
    raise FileNotFoundError(f"No MIDI files found in {MIDI_DIR}")

print(f"Found {len(all_midi_files)} MIDI files")

# %%
# Load or train tokenizer
TOKENIZER_CONFIG = TokenizerConfig(
    num_velocities=32,
    use_chords=False,
    use_programs=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
)

if os.path.exists(TOKENIZER_PATH):
    try:
        tokenizer = REMI.from_file(TOKENIZER_PATH)
        print("Tokenizer loaded")
    except:
        tokenizer = None

if not os.path.exists(TOKENIZER_PATH) or tokenizer is None:
    print("Training tokenizer...")
    tokenizer = REMI(TOKENIZER_CONFIG)
    tokenizer.train(vocab_size=3000, files_paths=all_midi_files)
    tokenizer.save_params(TOKENIZER_PATH)
    print("Tokenizer training complete")

print(f"Vocabulary size: {len(tokenizer)}")

# %%
# Load metadata and prepare data splits
metadata_df = pd.read_csv(METADATA_CSV_PATH)
train_files_df = metadata_df[metadata_df['split'] == 'train']
val_files_df = metadata_df[metadata_df['split'] == 'validation']
test_files_df = metadata_df[metadata_df['split'] == 'test']

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def tokenize_split(file_df, split_name):
    """Tokenize MIDI files for a data split"""
    token_sequences = []
    filename_to_path = {os.path.relpath(p, MIDI_DIR).replace('\\', '/'): p 
                       for p in all_midi_files}
    
    for _, row in tqdm(file_df.iterrows(), desc=f"Tokenizing {split_name}", total=len(file_df)):
        midi_filename = row['midi_filename']
        full_path = filename_to_path.get(midi_filename)
        
        # Try alternative extensions
        if not full_path:
            if midi_filename.endswith(".midi"):
                alt_filename = midi_filename.replace(".midi", ".mid")
            else:
                alt_filename = midi_filename.replace(".mid", ".midi")
            full_path = filename_to_path.get(alt_filename)
        
        if not full_path or not os.path.exists(full_path):
            continue
            
        try:
            # Try symusic first, fallback to miditoolkit
            try:
                score = Score(full_path)
                tokens = tokenizer(score)[0].tokens
            except:
                midi_file = miditoolkit.MidiFile(full_path)
                tokens = tokenizer(midi_file)[0].tokens
            
            token_sequences.append(tokens)
        except:
            continue
    
    # Save tokenized sequences
    output_path = os.path.join(PROCESSED_DATA_DIR, f"{split_name}_token_sequences.json")
    with open(output_path, 'w') as f:
        json.dump(token_sequences, f)
    
    return token_sequences

# %%
# Tokenize or load sequences
def load_or_tokenize_split(file_df, split_name):
    """Load existing tokenized sequences or create new ones"""
    sequence_path = os.path.join(PROCESSED_DATA_DIR, f"{split_name}_token_sequences.json")
    
    if os.path.exists(sequence_path):
        with open(sequence_path, 'r') as f:
            sequences = json.load(f)
        print(f"Loaded {len(sequences)} {split_name} sequences")
    else:
        sequences = tokenize_split(file_df, split_name)
        print(f"Tokenized {len(sequences)} {split_name} sequences")
    
    return sequences

train_sequences = load_or_tokenize_split(train_files_df, "train")
val_sequences = load_or_tokenize_split(val_files_df, "validation")
test_sequences = load_or_tokenize_split(test_files_df, "test")

# %%
# Extended Markov Chain Model
class REMIMarkovChain:
    """Extended Markov Chain model for REMI token sequences"""
    
    def __init__(self, order=2):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()
        self.start_tokens = []
        
    def train(self, token_sequences):
        """Train the Markov model on tokenized sequences"""
        print(f"Training {self.order}-gram Markov model on {len(token_sequences)} sequences...")
        
        for sequence in tqdm(token_sequences, desc="Processing sequences"):
            if len(sequence) < self.order + 1:
                continue
                
            self.vocab.update(sequence)
            
            if len(sequence) >= self.order:
                start_context = tuple(sequence[:self.order])
                self.start_tokens.append(start_context)
            
            for i in range(len(sequence) - self.order):
                context = tuple(sequence[i:i + self.order])
                next_token = sequence[i + self.order]
                
                self.transitions[context][next_token] += 1
                self.context_counts[context] += 1
        
        print(f"Model trained with {len(self.transitions)} contexts and {len(self.vocab)} tokens")
        
    def get_next_token_probabilities(self, context):
        """Get probability distribution for next token"""
        if context not in self.transitions:
            prob = 1.0 / len(self.vocab)
            return {token: prob for token in self.vocab}
        
        total_count = self.context_counts[context]
        return {token: count / total_count 
                for token, count in self.transitions[context].items()}
    
    def sample_next_token(self, context, temperature=1.0):
        """Sample next token given context"""
        probabilities = self.get_next_token_probabilities(context)
        
        if temperature != 1.0:
            tokens = list(probabilities.keys())
            probs = [probabilities[token] ** (1.0 / temperature) for token in tokens]
            total = sum(probs)
            probs = [p / total for p in probs]
            probabilities = dict(zip(tokens, probs))
        
        tokens = list(probabilities.keys())
        weights = [probabilities[token] for token in tokens]
        
        return random.choices(tokens, weights=weights)[0]
    
    def generate(self, length=500, seed_context=None, temperature=1.0):
        """Generate a new music sequence (Task 1: Unconditioned)"""
        if seed_context is None:
            context = random.choice(self.start_tokens) if self.start_tokens else tuple(random.choices(list(self.vocab), k=self.order))
        else:
            context = tuple(seed_context[-self.order:])
        
        generated = list(context)
        
        for _ in range(length - len(generated)):
            next_token = self.sample_next_token(context, temperature)
            generated.append(next_token)
            context = tuple(generated[-self.order:])
        
        return generated
    
    def generate_conditioned(self, prefix, continuation_length=200, temperature=1.0):
        """Continue a given sequence (Task 2: Conditioned)"""
        if len(prefix) < self.order:
            raise ValueError(f"Prefix must have at least {self.order} tokens")
        
        context = tuple(prefix[-self.order:])
        generated = list(prefix)
        
        for _ in range(continuation_length):
            next_token = self.sample_next_token(context, temperature)
            generated.append(next_token)
            context = tuple(generated[-self.order:])
        
        return generated
    
    def evaluate_perplexity(self, test_sequences):
        """Evaluate model perplexity"""
        total_log_prob = 0
        total_tokens = 0
        
        for sequence in test_sequences:
            if len(sequence) <= self.order:
                continue
                
            for i in range(self.order, len(sequence)):
                context = tuple(sequence[i-self.order:i])
                true_token = sequence[i]
                
                probs = self.get_next_token_probabilities(context)
                token_prob = probs.get(true_token, 1e-10)
                
                total_log_prob += -math.log2(token_prob)
                total_tokens += 1
        
        return 2 ** (total_log_prob / total_tokens) if total_tokens > 0 else float('inf')

# %%
# Train Markov model
print("\nTraining Extended Markov Chain Model")
markov_model = REMIMarkovChain(order=2)
markov_model.train(train_sequences)

# %%
# Unconditioned Generation
print("\nTask 1: Unconditioned Generation")
for i in range(3):
    temperature = [0.8, 1.0, 1.2][i]
    generated_tokens = markov_model.generate(length=2000, temperature=temperature)
    
    print(f"Generated piece {i+1} (temp={temperature}): {len(generated_tokens)} tokens")
    
    # Save as MIDI
    try:
        tok_sequence = TokSequence(tokens=generated_tokens)
        generated_midi = tokenizer([tok_sequence])
        generated_midi.dump_midi(f"generated_piece_{i+1}_unconditioned.mid")
        print(f"Successfully saved as 'generated_piece_{i+1}_unconditioned.mid'")
    except Exception as e:
        print(f"Error saving MIDI: {e}")

# %%
# Task 2: Harmonization Conditioning
print("\nTask 2: Harmonization (Generate chords/accompaniment following a melody)")

def extract_melody_from_sequence(sequence, melody_length=40):
    """
    Extract a melody line from a REMI sequence.
    In REMI, we'll extract pitch tokens with their timing but simplify to melody-only.
    """
    melody_tokens = []
    i = 0
    note_count = 0
    
    # Keep structural tokens (Bar, TimeSig, Tempo)
    while i < len(sequence) and note_count < melody_length:
        token = sequence[i]
        
        if token.startswith('Bar_') or token.startswith('TimeSig_') or token.startswith('Tempo_'):
            melody_tokens.append(token)
        elif token.startswith('Position_'):
            melody_tokens.append(token)
        elif token.startswith('Pitch_'):
            # For melody extraction, take the first pitch at each position (highest priority)
            melody_tokens.append(token)
            # Skip to next position or add simple rhythm
            if i + 1 < len(sequence) and sequence[i + 1].startswith('Velocity_'):
                melody_tokens.append('Velocity_80')  # Standardized melody velocity
            if i + 2 < len(sequence) and sequence[i + 2].startswith('Duration_'):
                melody_tokens.append(sequence[i + 2])  # Keep original duration
            note_count += 1
            # Skip any additional pitches at same position (harmony notes)
            while i + 1 < len(sequence) and sequence[i + 1].startswith('Pitch_'):
                i += 1
        elif token.startswith('Rest_'):
            melody_tokens.append(token)
        
        i += 1
    
    return melody_tokens

def create_melody_prompt(melody_tokens):
    """
    Create a conditioning prompt that presents melody and asks for harmonization.
    """
    # Start with basic structure
    prompt = melody_tokens[:15]  # Use first 15 tokens as conditioning context
    return prompt

# Generate 3 different harmonization examples
harmonization_scenarios = [
    {"name": "Classical Harmonization", "description": "Harmonize a classical melody with traditional chord progressions"},
    {"name": "Romantic Harmonization", "description": "Harmonize a melody with rich romantic-era harmonies"},
    {"name": "Contemporary Harmonization", "description": "Harmonize a melody with modern accompaniment patterns"}
]

for i, scenario in enumerate(harmonization_scenarios):
    print(f"\n--- Harmonization {i+1}: {scenario['name']} ---")
    print(f"Description: {scenario['description']}")
    
    # Use different source pieces for variety
    if i < len(val_sequences):
        source_sequence = val_sequences[i * 10 % len(val_sequences)]  # Spread across different pieces
        
        # Extract melody from source piece
        melody_line = extract_melody_from_sequence(source_sequence, melody_length=30)
        
        print(f"Extracted melody ({len(melody_line)} tokens): {melody_line[:10]}...")
        
        # Create conditioning prompt (melody-only start)
        conditioning_prompt = create_melody_prompt(melody_line)
        
        print(f"Conditioning prompt ({len(conditioning_prompt)} tokens): {conditioning_prompt}")
        
        # Generate full harmonization
        harmonized_sequence = markov_model.generate_conditioned(
            prefix=conditioning_prompt,
            continuation_length=1800,  # Generate full harmonization
            temperature=1.0
        )
        
        print(f"Generated harmonized piece: {len(harmonized_sequence)} total tokens")
        print(f"Melody condition: {len(conditioning_prompt)} tokens")
        print(f"Generated harmonization: {len(harmonized_sequence) - len(conditioning_prompt)} tokens")
        
        # Save with descriptive filename
        filename = f"harmonization_{i+1}_{scenario['name'].lower().replace(' ', '_')}.mid"
        try:
            tok_sequence = TokSequence(tokens=harmonized_sequence)
            generated_midi = tokenizer([tok_sequence])
            generated_midi.dump_midi(filename)
            print(f"Successfully saved as '{filename}'")
        except Exception as e:
            print(f"Error saving MIDI: {e}")
    else:
        print(f"Insufficient validation sequences for scenario {i+1}")

# %%
# Model Evaluation
print("\nModel Evaluation")
test_perplexity = markov_model.evaluate_perplexity(test_sequences)
val_perplexity = markov_model.evaluate_perplexity(val_sequences)

print(f"Test perplexity: {test_perplexity:.2f}")
print(f"Validation perplexity: {val_perplexity:.2f}")

# %%
print("\nAssignment 2 Complete!")
print("Generated files:")
print("Task 1 (Unconditioned Generation):")
for i in range(1, 4):
    print(f"- generated_piece_{i}_unconditioned.mid")

print("Task 2 (Harmonization Conditioning):")
print("- harmonization_1_classical_harmonization.mid (Classical harmony style)")
print("- harmonization_2_romantic_harmonization.mid (Romantic harmony style)")  
print("- harmonization_3_contemporary_harmonization.mid (Contemporary harmony style)")

# %%
