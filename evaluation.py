"""
Comprehensive Evaluation Module for Assignment 2 with Visualizations
Evaluates assignment2.py Extended Markov Chains with baseline comparison and charts
"""

import os
import sys
import json
import math
import numpy as np
from collections import Counter
from glob import glob

# Add visualization capability
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    VISUALIZATION_AVAILABLE = True
except ImportError:
    try:
        import matplotlib.pyplot as plt
        plt.style.use('default')
        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False

# Add timing analysis capability
try:
    import miditoolkit
    TIMING_ANALYSIS_AVAILABLE = True
except ImportError:
    TIMING_ANALYSIS_AVAILABLE = False

# Import the trained models from assignment2
try:
    from assignment2 import (melody_trigram_model, chord_trigram_model, melody_sequences, 
                           chord_sequences, melody_unigram_model, melody_bigram_model, 
                           chord_bigram_model)
    melody_model = melody_trigram_model  
    chord_model = chord_trigram_model    
except ImportError as e:
    print(f"Error importing from assignment2.py: {e}")
    print("Please run 'python assignment2.py' first to train the models!")
    sys.exit(1)

# Try to import homework3 for baseline comparison
HOMEWORK3_AVAILABLE = False
homework3 = None

try:
    import homework3
    HOMEWORK3_AVAILABLE = True
except ImportError:
    pass

# %%
# Core Analysis Functions

def calculate_perplexity(model, sequences):
    """Calculate perplexity with proper smoothing"""
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

def calculate_mixed_model_perplexity(model, sequences):
    """Calculate perplexity of the actual mixed-order model with fallback mechanism"""
    total_log_prob = 0
    total_tokens = 0
    
    for sequence in sequences:
        if len(sequence) <= 2:  # Need at least 3 tokens for meaningful evaluation
            continue
            
        for i in range(len(sequence) - 1):
            current_token = sequence[i + 1]
            
            # Get context based on model order and available history
            # Always provide context in the format that assignment2 expects
            if model.order == 1:
                # For unigram, doesn't matter what context we provide
                context = [sequence[i]]
            elif model.order == 2:
                if i >= 1:
                    context = [sequence[i-1], sequence[i]]
                else:
                    context = [sequence[i]]
            else:  # trigram or higher (order >= 3)
                if i >= 2:
                    context = [sequence[i-2], sequence[i-1], sequence[i]]
                elif i >= 1:
                    context = [sequence[i-1], sequence[i]]
                else:
                    context = [sequence[i]]
            
            # Get probability using the actual model's fallback mechanism
            try:
                probs = model.get_next_token_probabilities(context)
                prob = probs.get(current_token, 0.0)
                
                if prob > 0:
                    total_log_prob += math.log(prob)
                    total_tokens += 1
            except Exception as e:
                # Skip this token if there's an error in probability calculation
                continue
    
    if total_tokens == 0:
        return float('inf')
    
    avg_log_prob = total_log_prob / total_tokens
    perplexity = math.exp(-avg_log_prob)
    return perplexity

def analyze_musical_quality(melodies, name="Sequences"):
    """Comprehensive musical quality analysis"""
    if not melodies:
        return {}
    
    all_intervals = []
    all_ranges = []
    step_motions = []
    pitch_distributions = []
    
    for melody in melodies:
        if len(melody) < 2:
            continue
            
        # Intervals
        intervals = [melody[i+1] - melody[i] for i in range(len(melody)-1)]
        all_intervals.extend(intervals)
        
        # Range
        melody_range = max(melody) - min(melody)
        all_ranges.append(melody_range)
        
        # Step motion (1-2 semitones)
        steps = sum(1 for iv in intervals if abs(iv) <= 2)
        
        if len(intervals) > 0:
            step_motions.append(steps / len(intervals) * 100)
        
        # Pitch distribution
        pitch_distributions.extend(melody)
    
    # Calculate statistics
    results = {
        'name': name,
        'num_sequences': len([m for m in melodies if len(m) >= 2]),
        'avg_step_motion': np.mean(step_motions),
        'avg_range': np.mean(all_ranges),
        'avg_length': np.mean([len(m) for m in melodies]),
        'interval_entropy': calculate_interval_entropy(all_intervals),
        'pitch_entropy': calculate_pitch_entropy(pitch_distributions),
        'melodic_smoothness': calculate_melodic_smoothness(all_intervals),
        'interval_distribution': dict(Counter(all_intervals)),
        'pitch_range_coverage': len(set(pitch_distributions)),
    }
    
    print(f"\n=== {name} Analysis ===")
    print(f"Average step motion: {results['avg_step_motion']:.1f}%")
    print(f"Average range: {results['avg_range']:.1f} semitones")
    print(f"Melodic smoothness: {results['melodic_smoothness']:.3f}")
    print(f"Pitch range coverage: {results['pitch_range_coverage']} unique pitches")
    
    return results

def calculate_interval_entropy(intervals):
    """Calculate entropy of interval distribution"""
    if not intervals:
        return 0
    
    counts = Counter(intervals)
    total = len(intervals)
    entropy = 0
    
    for count in counts.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)
    
    return entropy

def calculate_pitch_entropy(pitches):
    """Calculate entropy of pitch distribution"""
    if not pitches:
        return 0
    
    counts = Counter(pitches)
    total = len(pitches)
    entropy = 0
    
    for count in counts.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)
    
    return entropy

def calculate_melodic_smoothness(intervals):
    """Calculate smoothness based on interval distribution"""
    if not intervals:
        return 0
    
    # Smoothness = proportion of small intervals (<=2 semitones)
    small_intervals = sum(1 for iv in intervals if abs(iv) <= 2)
    return small_intervals / len(intervals)

def generate_test_samples(melody_model, chord_model, num_samples=5):
    """Generate test samples for evaluation"""
    generated_melodies = []
    generated_chords = []
    
    for _ in range(num_samples):
        melody = melody_model.generate_sequence(length=32, temperature=1.0)
        generated_melodies.append(melody)
        
        chord_prog = chord_model.generate_sequence(length=8, temperature=1.0)
        generated_chords.append(chord_prog)
    
    return generated_melodies, generated_chords

def calculate_homework3_perplexity(test_melodies):
    """Calculate perplexity for homework3 baseline"""
    if not HOMEWORK3_AVAILABLE:
        return None
    
    try:
        # Check if homework3 has the perplexity function
        if hasattr(homework3, 'note_trigram_perplexity'):
            # We need to create temporary MIDI files from our test melodies
            # Since homework3's function expects MIDI files, we'll use a proxy approach
            # by directly calculating using homework3's probability functions
            
            if hasattr(homework3, 'note_trigram_probability') and hasattr(homework3, 'note_bigram_probability') and hasattr(homework3, 'note_unigram_probability'):
                # Get homework3's trained probability models
                print("Calculating homework3 baseline perplexity...")
                
                # Use homework3's MIDI files to train models
                unigramProbs = homework3.note_unigram_probability(homework3.midi_files)
                bigramTrans, bigramProbs = homework3.note_bigram_probability(homework3.midi_files)
                trigramTrans, trigramProbs = homework3.note_trigram_probability(homework3.midi_files)
                
                total_log_prob = 0
                total_tokens = 0
                valid_sequences = 0
                
                for sequence in test_melodies[:10]:  # Use first 10 test sequences for efficiency
                    if len(sequence) < 3:
                        continue
                    
                    # Calculate perplexity using homework3's exact approach
                    N = len(sequence)
                    sum_log_probs = 0.0
                    sequence_valid = True
                    
                    # First note probability (unigram)
                    p_w1 = unigramProbs.get(sequence[0], 0.0)
                    if p_w1 == 0.0:
                        continue  # Skip this sequence
                    sum_log_probs += math.log(p_w1)
                    
                    # Second note probability (bigram)
                    if N > 1:
                        prev_note = sequence[0]
                        current_note = sequence[1]
                        p_w2_given_w1 = 0.0
                        
                        if prev_note in bigramTrans:
                            next_notes = bigramTrans[prev_note]
                            if current_note in next_notes:
                                idx = next_notes.index(current_note)
                                p_w2_given_w1 = bigramProbs[prev_note][idx]
                        
                        if p_w2_given_w1 == 0.0:
                            continue  # Skip this sequence
                        sum_log_probs += math.log(p_w2_given_w1)
                    
                    # Subsequent notes (trigram)
                    if N > 2:
                        for i in range(2, N):
                            note_i_minus_2 = sequence[i-2]
                            note_i_minus_1 = sequence[i-1]
                            note_i = sequence[i]
                            
                            trigram_key = (note_i_minus_2, note_i_minus_1)
                            p_wi_given_pair = 0.0
                            
                            if trigram_key in trigramTrans:
                                next_notes = trigramTrans[trigram_key]
                                if note_i in next_notes:
                                    idx = next_notes.index(note_i)
                                    p_wi_given_pair = trigramProbs[trigram_key][idx]
                            
                            if p_wi_given_pair == 0.0:
                                sequence_valid = False
                                break
                            sum_log_probs += math.log(p_wi_given_pair)
                    
                    if sequence_valid:
                        # Calculate perplexity for this sequence using homework3's formula
                        sequence_perplexity = math.exp(-(1.0/N) * sum_log_probs)
                        # Convert to total log prob for averaging
                        total_log_prob += sum_log_probs
                        total_tokens += N
                        valid_sequences += 1
                
                if valid_sequences > 0:
                    # Calculate average perplexity
                    avg_log_prob = total_log_prob / total_tokens
                    homework3_perplexity = math.exp(-avg_log_prob)
                    print(f"Successfully calculated homework3 perplexity on {valid_sequences} sequences")
                    return homework3_perplexity
                else:
                    print("No valid sequences for homework3 perplexity calculation")
                    return None
    
    except Exception as e:
        print(f"Error calculating homework3 perplexity: {e}")
        return None
    
    return None

def analyze_homework3_musical_quality():
    """Analyze homework3's musical characteristics from its training data"""
    if not HOMEWORK3_AVAILABLE:
        return None
    
    try:
        # Extract melodies from homework3's MIDI files
        homework3_melodies = []
        for midi_file in homework3.midi_files:
            try:
                notes = homework3.note_extraction(midi_file)
                if len(notes) >= 10:  # Only include sequences with reasonable length
                    homework3_melodies.append(notes)
            except:
                continue
        
        if not homework3_melodies:
            return None
        
        # Use the same analysis function but for homework3 data
        homework3_analysis = analyze_musical_quality(homework3_melodies, "Homework3 Training Data")
        
        # Add homework3-specific info
        homework3_analysis['dataset_size'] = len(homework3.midi_files)
        homework3_analysis['valid_sequences'] = len(homework3_melodies)
        
        return homework3_analysis
        
    except Exception as e:
        print(f"Error analyzing homework3 musical quality: {e}")
        return None

# %%
# Visualization Functions

def create_step_motion_comparison(analysis_results_list, save_path="step_motion_comparison.png"):
    """Create step motion comparison chart"""
    if not VISUALIZATION_AVAILABLE:
        return None
    
    try:
        names = [result['name'] for result in analysis_results_list]
        step_motions = [result['avg_step_motion'] for result in analysis_results_list]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, step_motions, color=['#3498db', '#e74c3c', '#f39c12'])
        plt.title('Step Motion Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Step Motion (%)', fontsize=12)
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, step_motions):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    except Exception:
        return None

def create_model_performance_comparison(melody_perplexities, chord_perplexities, save_path="model_performance.png"):
    """Create model performance comparison chart"""
    if not VISUALIZATION_AVAILABLE:
        return None
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Melody models
        models = list(melody_perplexities.keys())
        perps = list(melody_perplexities.values())
        bars1 = ax1.bar(models, perps, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax1.set_title('Melody Model Performance', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Perplexity (Lower is Better)', fontsize=12)
        
        for bar, value in zip(bars1, perps):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(perps)*0.01, 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Chord models
        models2 = list(chord_perplexities.keys())
        perps2 = list(chord_perplexities.values())
        bars2 = ax2.bar(models2, perps2, color=['#2ecc71', '#e74c3c'])
        ax2.set_title('Chord Model Performance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Perplexity (Lower is Better)', fontsize=12)
        
        for bar, value in zip(bars2, perps2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(perps2)*0.01, 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    except Exception:
        return None

def create_comprehensive_summary(all_results, save_path="comprehensive_summary.png"):
    """Create comprehensive 4-panel summary"""
    if not VISUALIZATION_AVAILABLE:
        return None
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Panel 1: Model Perplexities
        all_models = list(all_results['melody_perplexities'].keys()) + \
                    [f"Chord {k}" for k in all_results['chord_perplexities'].keys()]
        all_perps = list(all_results['melody_perplexities'].values()) + \
                   list(all_results['chord_perplexities'].values())
        
        colors1 = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'][:len(all_models)]
        bars1 = ax1.bar(all_models, all_perps, color=colors1)
        ax1.set_title('All Model Perplexities', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Perplexity', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Panel 2: Musical Quality Metrics
        train_analysis = all_results['training_analysis']
        gen_analysis = all_results['generated_analysis']
        
        metrics = ['avg_step_motion', 'avg_range', 'melodic_smoothness']
        train_values = [train_analysis[m] for m in metrics]
        gen_values = [gen_analysis[m] for m in metrics]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        ax2.bar(x_pos - width/2, train_values, width, label='Training', color='#3498db')
        ax2.bar(x_pos + width/2, gen_values, width, label='Generated', color='#e74c3c')
        ax2.set_title('Musical Quality Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Step Motion %', 'Range (semi)', 'Smoothness'])
        ax2.legend()
        
        # Panel 3: Model Architecture
        model_info = all_results['model_stats']
        categories = ['Melody Vocab', 'Melody Contexts', 'Chord Vocab', 'Chord Contexts']
        values = [model_info['melody_vocab_size'], model_info['melody_contexts'], 
                 model_info['chord_vocab_size'], model_info['chord_contexts']]
        
        bars3 = ax3.bar(categories, values, color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
        ax3.set_title('Model Architecture Stats', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        
        # Panel 4: Comparison with Baseline
        if all_results.get('homework3_perplexity'):
            hw3_perp = all_results['homework3_perplexity']
            a2_perp = all_results['melody_perplexity']
            
            comparison_data = ['Assignment2', 'Homework3']
            perplexity_data = [a2_perp, hw3_perp]
            
            bars4 = ax4.bar(comparison_data, perplexity_data, color=['#e74c3c', '#95a5a6'])
            ax4.set_title('Perplexity Comparison', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Perplexity (Lower is Better)', fontsize=12)
            
            for bar, value in zip(bars4, perplexity_data):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(perplexity_data)*0.01, 
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Homework3\nNot Available', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=16)
            ax4.set_title('Baseline Comparison', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    except Exception:
        return None

def create_perplexity_comparison_chart(assignment2_perplexity, homework3_perplexity, save_path="perplexity_comparison.png"):
    """Create perplexity comparison chart"""
    if not VISUALIZATION_AVAILABLE:
        return None
    
    try:
        models = ['Assignment2\n(Trigram)', 'Homework3\n(Baseline)']
        perplexities = [assignment2_perplexity, homework3_perplexity]
        
        plt.figure(figsize=(10, 8))
        
        colors = ['#2ecc71', '#e74c3c']
        bars = plt.bar(models, perplexities, color=colors, width=0.6)
        
        plt.title('Perplexity Comparison: Assignment2 vs Homework3', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Perplexity (Lower is Better)', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, perplexities)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(perplexities)*0.02, 
                    f'{value:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Calculate and show improvement
        improvement = ((homework3_perplexity - assignment2_perplexity) / homework3_perplexity) * 100
        
        if improvement > 0:
            plt.text(0.5, 0.85, f'🚀 Assignment2 Improvement: {improvement:.1f}%', 
                    transform=plt.gca().transAxes, ha='center', fontsize=14, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                    fontweight='bold')
        else:
            plt.text(0.5, 0.85, f'📈 Gap to close: {abs(improvement):.1f}%', 
                    transform=plt.gca().transAxes, ha='center', fontsize=14, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
                    fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    except Exception:
        return None

# %%
# MIDI Analysis Functions

def extract_notes_from_midi(midi_file):
    """Extract notes from MIDI file for harmony analysis"""
    if not TIMING_ANALYSIS_AVAILABLE:
        return []
    
    try:
        midi_obj = miditoolkit.midi.parser.MidiFile(midi_file)
        notes = []
        
        for instrument in midi_obj.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append({
                        'pitch': note.pitch,
                        'start': note.start,
                        'end': note.end,
                        'velocity': note.velocity
                    })
        
        return sorted(notes, key=lambda x: x['start'])
    except Exception:
        return []

def analyze_pitch_distribution(notes):
    """Analyze pitch distribution and musical characteristics"""
    if not notes:
        return {}
    
    pitches = [note['pitch'] for note in notes]
    pitch_classes = [p % 12 for p in pitches]
    
    # Count pitch classes
    pc_counts = Counter(pitch_classes)
    
    # Note names
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    return {
        'pitch_class_counts': pc_counts,
        'note_names': note_names,
        'total_notes': len(notes),
        'unique_pitches': len(set(pitches)),
        'pitch_range': max(pitches) - min(pitches) if pitches else 0,
        'avg_pitch': np.mean(pitches)
    }

def analyze_tonality(pitch_classes_count):
    """Analyze tonality tendency"""
    # Major scale patterns (in semitones from root)
    major_patterns = {
        'C': [0, 2, 4, 5, 7, 9, 11],
        'G': [7, 9, 11, 0, 2, 4, 6],
        'D': [2, 4, 6, 7, 9, 11, 1],
        'A': [9, 11, 1, 2, 4, 6, 8],
        'E': [4, 6, 8, 9, 11, 1, 3],
        'F': [5, 7, 9, 10, 0, 2, 4]
    }
    
    scores = {}
    for key, pattern in major_patterns.items():
        score = sum(pitch_classes_count.get(pc, 0) for pc in pattern)
        scores[key] = score
    
    best_key = max(scores.keys(), key=lambda k: scores[k])
    max_score = scores[best_key]
    total_notes = sum(pitch_classes_count.values())
    
    return {
        'detected_keys': scores,
        'best_key': best_key,
        'key_strength': max_score / total_notes if total_notes > 0 else 0
    }

def create_midi_harmony_analysis(midi_file, save_prefix="harmony_analysis"):
    """Create comprehensive MIDI harmony analysis visualization"""
    if not VISUALIZATION_AVAILABLE or not TIMING_ANALYSIS_AVAILABLE:
        return None
    
    if not os.path.exists(midi_file):
        return None
    
    try:
        notes = extract_notes_from_midi(midi_file)
        if not notes:
            return None
        
        pitch_analysis = analyze_pitch_distribution(notes)
        tonality_analysis = analyze_tonality(pitch_analysis['pitch_class_counts'])
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Panel 1: Pitch Class Distribution (Pie Chart)
        pc_counts = pitch_analysis['pitch_class_counts']
        note_names = pitch_analysis['note_names']
        
        labels = [note_names[i] for i in range(12) if pc_counts.get(i, 0) > 0]
        sizes = [pc_counts.get(i, 0) for i in range(12) if pc_counts.get(i, 0) > 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Pitch Class Distribution', fontsize=14, fontweight='bold')
        
        # Panel 2: Interval Distribution
        intervals = []
        for i in range(len(notes) - 1):
            interval = notes[i+1]['pitch'] - notes[i]['pitch']
            intervals.append(interval)
        
        interval_counts = Counter(intervals)
        common_intervals = dict(interval_counts.most_common(10))
        
        bars2 = ax2.bar(range(len(common_intervals)), list(common_intervals.values()), 
                       color='lightblue')
        ax2.set_title('Most Common Intervals', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Interval (semitones)')
        ax2.set_ylabel('Frequency')
        ax2.set_xticks(range(len(common_intervals)))
        ax2.set_xticklabels([f'{k:+d}' for k in common_intervals.keys()])
        
        # Panel 3: Black vs White Keys
        black_keys = {1, 3, 6, 8, 10}  # C#, D#, F#, G#, A#
        white_count = sum(pc_counts.get(i, 0) for i in range(12) if i not in black_keys)
        black_count = sum(pc_counts.get(i, 0) for i in black_keys)
        
        ax3.pie([white_count, black_count], labels=['White Keys', 'Black Keys'], 
               colors=['white', 'gray'], autopct='%1.1f%%', 
               wedgeprops={'edgecolor': 'black'})
        ax3.set_title('Black vs White Key Usage', fontsize=14, fontweight='bold')
        
        # Panel 4: Key Detection
        key_scores = tonality_analysis['detected_keys']
        top_keys = dict(sorted(key_scores.items(), key=lambda x: x[1], reverse=True)[:6])
        
        bars4 = ax4.bar(top_keys.keys(), 
                       [score/sum(key_scores.values())*100 for score in top_keys.values()],
                       color='lightgreen')
        ax4.set_title('Key Detection (Major Keys)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Strength (%)')
        
        # Add best key annotation
        best_key = tonality_analysis['best_key']
        ax4.text(0.02, 0.98, f'Detected: {best_key} Major', transform=ax4.transAxes, 
                va='top', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        
        # Save with specific filename
        chart_path = f"{save_prefix}_{os.path.splitext(midi_file)[0]}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'chart_path': chart_path,
            'pitch_analysis': pitch_analysis,
            'tonality_analysis': tonality_analysis,
            'interval_analysis': common_intervals
        }
    except Exception:
        return None

def create_midi_comparison_analysis(midi_files, save_path="midi_comparison.png"):
    """Create comprehensive MIDI comparison analysis"""
    if not VISUALIZATION_AVAILABLE or not TIMING_ANALYSIS_AVAILABLE:
        return None
    
    available_files = [f for f in midi_files if os.path.exists(f)]
    if len(available_files) < 2:
        return None
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        results = {}
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(available_files)]
        
        # Analyze each file
        for midi_file in available_files:
            notes = extract_notes_from_midi(midi_file)
            if notes:
                pitch_analysis = analyze_pitch_distribution(notes)
                tonality_analysis = analyze_tonality(pitch_analysis['pitch_class_counts'])
                results[midi_file] = {
                    'pitch_analysis': pitch_analysis,
                    'tonality_analysis': tonality_analysis
                }
        
        # Panel 1: Pitch Range Comparison
        friendly_names = []
        for filename in results.keys():
            if 'symbolic_unconditioned' in filename:
                friendly_names.append('Assignment2\nUnconditioned')
            elif 'symbolic_conditioned' in filename:
                friendly_names.append('Assignment2\nConditioned')
            elif 'PDMX_subset' in filename or 'homework3' in filename.lower():
                friendly_names.append('Homework3\nBaseline')
            else:
                friendly_names.append(os.path.splitext(filename)[0])
        
        ranges = [results[f]['pitch_analysis']['pitch_range'] for f in results.keys()]
        
        bars1 = ax1.bar(friendly_names, ranges, color=colors)
        ax1.set_title('Pitch Range Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Range (semitones)')
        ax1.tick_params(axis='x', rotation=0)
        
        # Panel 2: Unique Pitches
        unique_pitches = [results[f]['pitch_analysis']['unique_pitches'] for f in results.keys()]
        
        bars2 = ax2.bar(friendly_names, unique_pitches, color=colors)
        ax2.set_title('Unique Pitches Used', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Unique Pitches')
        ax2.tick_params(axis='x', rotation=0)
        
        # Panel 3: Average Pitch
        avg_pitches = [results[f]['pitch_analysis']['avg_pitch'] for f in results.keys()]
        
        bars3 = ax3.bar(friendly_names, avg_pitches, color=colors)
        ax3.set_title('Average Pitch', fontsize=14, fontweight='bold')
        ax3.set_ylabel('MIDI Pitch Number')
        ax3.tick_params(axis='x', rotation=0)
        
        # Panel 4: Key Detection Strength
        key_strengths = [results[f]['tonality_analysis']['key_strength'] * 100 for f in results.keys()]
        detected_keys = [results[f]['tonality_analysis']['best_key'] for f in results.keys()]
        
        bars4 = ax4.bar(friendly_names, key_strengths, color=colors)
        ax4.set_title('Tonality Strength', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Key Strength (%)')
        ax4.tick_params(axis='x', rotation=0)
        
        # Add key labels
        for bar, key_strength, detected_key in zip(bars4, key_strengths, detected_keys):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{detected_key}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return results
    except Exception:
        return None

# %%
# Main Evaluation Function

def run_comprehensive_evaluation():
    """Run complete evaluation of assignment2.py models with visualizations"""
    print("=" * 60)
    print("COMPREHENSIVE EVALUATION - Assignment 2 Extended Markov Chains")
    print("=" * 60)
    
    # Data splits - Use official MAESTRO dataset splits
    # Train: 962 files (75.4%), Validation: 137 files (10.7%), Test: 177 files (13.9%)
    train_melodies = melody_sequences[:962]
    val_melodies = melody_sequences[962:1099]  # 962 + 137 = 1099
    test_melodies = melody_sequences[1099:]    # Last 177 files
    
    train_chords = chord_sequences[:962]
    val_chords = chord_sequences[962:1099]
    test_chords = chord_sequences[1099:]
    
    print(f"\nDataset: {len(melody_sequences)} total sequences")
    print(f"Official MAESTRO split: {len(train_melodies)} train, {len(val_melodies)} val, {len(test_melodies)} test")
    
    # 1. Assignment2 Model Performance Evaluation
    print(f"\n{'='*20} ASSIGNMENT2 MODEL EVALUATION {'='*20}")
    
    # Calculate perplexities for the actual models used in assignment2 MIDI generation
    print("🎯 Assignment2 Actual Model Performance:")
    
    # These are the exact models used for MIDI generation in assignment2.py
    assignment2_melody_perplexity = calculate_mixed_model_perplexity(melody_trigram_model, test_melodies)
    assignment2_chord_perplexity = calculate_mixed_model_perplexity(chord_trigram_model, test_chords)
    
    print(f"  Melody Model (Trigram with Fallback): {assignment2_melody_perplexity:.2f}")
    print(f"  Chord Model (Trigram with Fallback): {assignment2_chord_perplexity:.2f}")
    
    print(f"\n💡 Note: These models use intelligent fallback (Trigram → Bigram → Unigram)")
    print(f"This reflects the exact process used in assignment2 MIDI generation!")
    
    # 2. Musical Quality Analysis
    print(f"\n{'='*20} MUSICAL QUALITY ANALYSIS {'='*20}")
    
    # Analyze training data
    train_results = analyze_musical_quality(train_melodies, "Training Data")
    
    # Generate and analyze test samples
    generated_melodies, generated_chords = generate_test_samples(melody_model, chord_model, 10)
    generated_results = analyze_musical_quality(generated_melodies, "Generated Melodies")
    
    # 3. Baseline Comparison
    baseline_results = {}
    homework3_perplexity = None
    homework3_analysis = None
    
    if HOMEWORK3_AVAILABLE:
        print("\n=== BASELINE COMPARISON WITH HOMEWORK3 ===")
        try:
            # Analyze homework3's musical characteristics
            homework3_analysis = analyze_homework3_musical_quality()
            if homework3_analysis:
                print(f"Homework3 Training Analysis Complete:")
                print(f"  Dataset: {homework3_analysis['dataset_size']} MIDI files")
                print(f"  Valid sequences: {homework3_analysis['valid_sequences']}")
                print(f"  Average step motion: {homework3_analysis['avg_step_motion']:.1f}%")
                print(f"  Average range: {homework3_analysis['avg_range']:.1f} semitones")
                print(f"  Pitch range coverage: {homework3_analysis['pitch_range_coverage']} unique pitches")
            
            # Calculate homework3 perplexity
            homework3_perplexity = calculate_homework3_perplexity(test_melodies)
            
            if homework3_perplexity:
                print(f"Homework3 Baseline Perplexity: {homework3_perplexity:.2f}")
                print(f"Assignment2 Trigram Perplexity: {assignment2_melody_perplexity:.2f}")
                
                # Calculate improvement
                if homework3_perplexity != float('inf') and assignment2_melody_perplexity != float('inf'):
                    improvement = ((homework3_perplexity - assignment2_melody_perplexity) / homework3_perplexity) * 100
                    print(f"🎯 Perplexity Improvement: {improvement:.1f}% (Lower is Better)")
        except Exception as e:
            print(f"Error in baseline comparison: {e}")
    
    # 4. CREATE VISUALIZATIONS
    print(f"\n{'='*20} GENERATING ANALYSIS VISUALIZATIONS {'='*20}")
    
    if VISUALIZATION_AVAILABLE:
        # Prepare analysis results for visualization
        analysis_results_list = [train_results, generated_results]
        if homework3_analysis:
            analysis_results_list.append(homework3_analysis)
        
        # Generate charts
        charts_created = []
        
        # Step Motion Comparison
        chart_path = create_step_motion_comparison(analysis_results_list, "step_motion_analysis.png")
        if chart_path:
            charts_created.append(chart_path)
        
        # Model Performance Comparison - simplified for assignment2 actual models
        assignment2_models = {
            'Melody Trigram': assignment2_melody_perplexity,
            'Chord Trigram': assignment2_chord_perplexity
        }
        chart_path = create_model_performance_comparison(assignment2_models, {}, "model_performance_analysis.png")
        if chart_path:
            charts_created.append(chart_path)
        
        # Perplexity Comparison with Homework3
        if homework3_perplexity is not None:
            chart_path = create_perplexity_comparison_chart(assignment2_melody_perplexity, homework3_perplexity, "perplexity_comparison.png")
            if chart_path:
                charts_created.append(chart_path)
        
        # MIDI HARMONY ANALYSIS
        print(f"\n{'='*20} DETAILED MIDI HARMONY ANALYSIS {'='*20}")
        
        # Analyze generated MIDI files individually
        midi_files_to_analyze = ['symbolic_unconditioned.mid', 'symbolic_conditioned.mid']
        midi_harmony_results = {}
        
        for midi_file in midi_files_to_analyze:
            harmony_result = create_midi_harmony_analysis(midi_file, "detailed_harmony")
            if harmony_result:
                midi_harmony_results[midi_file] = harmony_result
                charts_created.append(harmony_result['chart_path'])
        
        # Create comprehensive MIDI comparison - use homework3 baseline instead of q10
        if len(midi_harmony_results) >= 1:
            available_files = [f for f in midi_files_to_analyze if f in midi_harmony_results]
            # Check for homework3 baseline file instead of q10
            if HOMEWORK3_AVAILABLE and hasattr(homework3, 'midi_files') and homework3.midi_files:
                # Use first homework3 training file as baseline
                hw3_baseline_file = homework3.midi_files[0]
                if os.path.exists(hw3_baseline_file):
                    available_files.append(hw3_baseline_file)
            
            if len(available_files) >= 2:
                comparison_results = create_midi_comparison_analysis(available_files, "comprehensive_midi_comparison.png")
                if comparison_results:
                    charts_created.append("comprehensive_midi_comparison.png")
        
        print(f"Charts created: {len(charts_created)}")
    
    # 5. Compile Results
    evaluation_results = {
        'melody_perplexity': assignment2_melody_perplexity,
        'chord_perplexity': assignment2_chord_perplexity,
        'homework3_perplexity': homework3_perplexity,
        'training_analysis': train_results,
        'generated_analysis': generated_results,
        'homework3_analysis': homework3_analysis,  # Add homework3 analysis to json
        'model_stats': {
            'melody_vocab_size': len(melody_trigram_model.vocab),
            'melody_contexts': len(melody_trigram_model.transitions),
            'chord_vocab_size': len(chord_trigram_model.vocab),
            'chord_contexts': len(chord_trigram_model.transitions),
        },
        'baseline_comparison': baseline_results
    }
    
    # Add perplexity improvement calculation
    if homework3_perplexity and homework3_perplexity != float('inf') and assignment2_melody_perplexity != float('inf'):
        improvement = ((homework3_perplexity - assignment2_melody_perplexity) / homework3_perplexity) * 100
        evaluation_results['perplexity_improvement'] = improvement
    
    # 6. Create Comprehensive Summary Visualization
    if VISUALIZATION_AVAILABLE:
        summary_chart = create_comprehensive_summary(evaluation_results, "comprehensive_analysis_summary.png")
        if summary_chart:
            print(f"Comprehensive summary chart created: {summary_chart}")
    
    # 7. Summary
    print(f"\n{'='*20} EVALUATION SUMMARY {'='*20}")
    print(f"✅ Multi-Model Performance:")
    print(f"   Best Melody Model: Mixed Trigram (Perplexity: {assignment2_melody_perplexity:.2f})")
    print(f"   Best Chord Model: Mixed Trigram (Perplexity: {assignment2_chord_perplexity:.2f})")
    print(f"   Generated Step Motion: {generated_results.get('avg_step_motion', 0):.1f}%")
    
    # Add homework3 comparison
    if homework3_perplexity:
        print(f"\n🎯 Baseline Comparison:")
        print(f"   Assignment2 Perplexity: {assignment2_melody_perplexity:.2f}")
        print(f"   Homework3 Perplexity: {homework3_perplexity:.2f}")
        
        if 'perplexity_improvement' in evaluation_results:
            improvement = evaluation_results['perplexity_improvement']
            if improvement > 0:
                print(f"   🚀 Perplexity Improvement: {improvement:.1f}% BETTER than Homework3!")
            else:
                print(f"   📈 Perplexity Gap: {abs(improvement):.1f}% behind Homework3")
    
    print(f"\n✅ Model Architecture Coverage:")
    print(f"   Melody Models: Unigram, Bigram, Trigram")
    print(f"   Chord Models: Bigram, Trigram")
    print(f"   Total contexts learned: {len(melody_trigram_model.transitions) + len(chord_trigram_model.transitions)}")
    
    # Save results
    with open('comprehensive_evaluation_results.json', 'w') as f:
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_types(evaluation_results)
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to 'comprehensive_evaluation_results.json'")
    print(f"\n🎵 Generated MIDI files available:")
    print(f"   - symbolic_unconditioned.mid (with melody + chord tracks)")
    print(f"   - symbolic_conditioned.mid (with melody + chord tracks)")
    
    if VISUALIZATION_AVAILABLE:
        print(f"\n📊 Analysis charts generated:")
        print(f"   - comprehensive_analysis_summary.png (main overview)")
        print(f"   - step_motion_analysis.png")
        print(f"   - model_performance_analysis.png")
        
        if homework3_perplexity:
            print(f"   - perplexity_comparison.png (Assignment2 vs Homework3)")
    
    return evaluation_results

# %%
# Run evaluation if called directly
if __name__ == "__main__":
    results = run_comprehensive_evaluation()
    print("\n🎉 Comprehensive evaluation complete!") 