# Chord Analysis Study - Complete MIDI Dataset Analysis

## Overview

This study analyzes the COMPLETE classical MIDI dataset to understand real harmonic progressions as they appear in actual classical music.

## Data Source

- **195 MIDI files** from `/data/raw/`
- **194 files successfully analyzed** (1 had issues)
- **59,657 chord events** extracted
- **12 composers** identified: Beethoven, Mozart, Chopin, Debussy, Bach, Schubert, Schumann, Grieg, Mendelssohn, Rachmaninoff, Liszt, and others

## Key Findings from Complete Dataset

### 1. Overall Chord Distribution

From 59,657 actual chord events:
- **60% Major chords** vs 40% Minor chords
- Classical music shows clear major tonality preference
- Average **307 chord changes per piece**

### 2. Most Common Chords Across All Files

| Chord | Occurrences | Percentage | Notes |
|-------|------------|------------|-------|
| A#/Bb major | 2,039 | 5.4% | Chromatic/modulation |
| C major | 2,001 | 5.3% | Tonic (in C) |
| D#/Eb major | 1,986 | 5.3% | Chromatic |
| F major | 1,902 | 5.0% | Subdominant (in C) |
| G major | 1,593 | 4.2% | Dominant (in C) |
| A major | 1,559 | 4.1% | Secondary dominant |
| D major | 1,536 | 4.1% | Secondary dominant |
| C minor | 1,453 | 3.9% | Parallel minor |
| A minor | 1,450 | 3.8% | Relative minor |
| E major | 1,367 | 3.6% | Mediant |

### 3. Composer-Specific Analysis

#### Beethoven (27 files, 14,122 chords)
- **56% major, 43% minor**
- Strong D#/Eb emphasis (4.5% of all chords)
- Classical clarity with frequent A#/Bb (modulation to subdominant)
- Average 523 chords per piece (longer works)

#### Mozart (21 files, 10,102 chords)
- **59% major, 33% minor, 6% other**
- F major dominates (6.2%) - subdominant emphasis
- C major (5.4%) - clear tonic
- Average 481 chords per piece

#### Chopin (47 files, 11,337 chords)
- **60% major, 32% minor, 6% other**
- G#/Ab major prominent (4.3%) - romantic chromaticism
- F#/Gb major (4.0%) - distant key relationships
- Average 241 chords per piece (shorter character pieces)

#### Debussy (8 files, 2,140 chords)
- **62% major, 20% minor, 16% other**
- Higher percentage of "other" qualities (sus chords, extended harmonies)
- D major (4.0%) - whole tone implications
- Impressionist harmony evident

#### Bach (3 files, 965 chords)
- **37% major, 47% minor, 14% other**
- ONLY composer with more minor than major!
- F minor prominent (6.8%)
- Baroque counterpoint creates different harmonic profile

#### Rachmaninoff (2 files, 230 chords)
- **23% major, 48% minor, 28% other**
- Heavily minor-oriented (Russian romanticism)
- C# minor dominant (11.7%)
- Rich extended harmonies (28% other)

#### Schubert (23 files, 3,390 chords)
- **56% major, 27% minor, 15% other**
- G minor prominent (7.2%)
- Known for major/minor mixture

#### Grieg (16 files, 3,516 chords)
- **67% major, 23% minor, 8% other**
- Most major-oriented composer
- Norwegian folk influence

### 4. Historical Period Trends

| Period | Composers | Major % | Minor % | Other % |
|--------|-----------|---------|---------|---------|
| Baroque | Bach | 37% | 47% | 14% |
| Classical | Mozart, Beethoven | 57% | 38% | 5% |
| Early Romantic | Schubert, Chopin, Schumann | 58% | 27% | 15% |
| Late Romantic | Liszt, Rachmaninoff | 30% | 41% | 27% |
| Impressionist | Debussy | 62% | 20% | 16% |
| Nationalist | Grieg | 67% | 23% | 8% |

### 5. Key Observations

1. **Chromatic notes (Bb, Eb) appear in top 3** - Shows extensive modulation in classical music
2. **Bach is outlier** - More minor than major (Baroque style)
3. **Rachmaninoff most minor-focused** - Russian romantic darkness
4. **Grieg most major-focused** - Folk influence
5. **Extended harmonies increase over time** - 5% (Classical) to 27% (Late Romantic)

## How to Use This Study

### Running the Analysis

```bash
# Extract chords from ALL MIDI files (takes ~5 minutes)
python3 midi_chord_extractor.py

# Generate interpretation and visualizations
python3 harmonic_pattern_analyzer.py
```

### Output Files

- `output/midi_chord_analysis.txt` - Raw statistics from 194 files
- `output/chord_analysis_data.json` - Complete JSON dataset
- `output/harmonic_interpretation.txt` - Musical interpretation
- `output/harmonic_analysis_charts.png` - Visual analysis by composer

## Technical Details

### Chord Extraction Method

1. **Sampling**: Every 480 ticks (quarter note)
2. **Chord Detection**: 2+ simultaneous notes required
3. **Template Matching**: 15 chord types (major, minor, dim, aug, 7ths, sus, etc.)
4. **Quality Detection**: Identifies inversions and extensions

### Pitch Class Mapping

```
0 = C    4 = E    8 = G#/Ab
1 = C#   5 = F    9 = A
2 = D    6 = F#   10 = A#/Bb
3 = D#   7 = G    11 = B
```

## Insights for Model Training

This analysis reveals critical patterns for training:

1. **Chromatic movement is essential** - Bb and Eb in top 3 shows modulation is key
2. **Composer style is measurable** - Clear statistical signatures per composer
3. **Historical evolution exists** - Complexity increases from Classical to Romantic
4. **Major bias but composer-specific** - Bach minor, Grieg major
5. **Average harmonic rhythm** - 307 chords/piece suggests sampling strategy

## What This Means

The data shows classical music is NOT just simple I-IV-V progressions. The prominence of chromatic chords (Bb, Eb) in the top 3 indicates:

- Frequent modulation between keys
- Secondary dominants are common
- Borrowed chords from parallel keys
- Rich harmonic vocabulary beyond diatonic

This real data from YOUR actual MIDI files provides ground truth for training models that can capture the true complexity of classical harmony.