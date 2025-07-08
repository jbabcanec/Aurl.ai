# 🎵 Aurl.ai MIDI Data Transformation Analysis

## 📊 Dataset Overview

We analyzed **10 classical piano pieces** including:
- Beethoven's Waldstein Sonata
- Chopin Preludes and Ballades 
- Mendelssohn's Songs Without Words
- Grieg's Lyric Pieces

### Key Statistics:
- **Total Notes**: 23,689
- **Files**: 10 classical piano pieces
- **Duration**: Average 6.8 minutes per piece
- **Polyphony**: Max 5 simultaneous notes observed (lower than expected 8-12 for complex piano)

## 🎹 Musical Characteristics

### Pitch Distribution
- **Range**: 24-101 (MIDI notes, covering A0 to F7)
- **Most Common Notes**:
  - E3 (52): 848 occurrences
  - A3 (57): 769 occurrences  
  - B3 (59): 754 occurrences
  - E4 (64): 741 occurrences
- **Average Pitch**: 62.6 (D4)

### Velocity (Dynamics) Analysis
- **Range**: 14-117 (ppp to fff)
- **Average**: 52.6 (medium soft)
- **Distribution**: Bell curve centered around medium velocities
  - Most notes in bins 2-4 (velocities 32-96)
  - Very few extreme soft or loud notes

### Note Durations
- **Range**: 0.01s to 6.27s
- **Average**: 0.26s (roughly 16th note at 120 BPM)
- **Distribution**:
  - Short (<0.25s): 17,693 notes (74.7%)
  - Medium (0.25-1s): 5,114 notes (21.6%)
  - Long (>1s): 882 notes (3.7%)

## 🔄 Data Transformation Process

### Example: Waldstein Sonata Movement 1

**Original MIDI**:
- Duration: 618.4 seconds (10.3 minutes)
- Notes: 8,576
- Instruments: 2 (Piano right & left hands)
- Note density: 13.87 notes/second

**After Transformation**:
- **Events**: 35,955 musical events
- **Tokens**: 35,955 tokens
- **Piano Roll**: 4,948 time steps × 88 pitches

### Token Sequence Example

Here's how the beginning of Waldstein transforms:

```
Position | Token | Event Type      | Value | Meaning
---------|-------|-----------------|-------|------------------
0        | 0     | START_TOKEN     | 0     | Start of piece
1        | 322   | VELOCITY_CHANGE | 17    | Set velocity ~68
2        | 45    | NOTE_ON         | 62    | Play D4
3        | 181   | TIME_SHIFT      | 1     | Wait 125ms
4        | 133   | NOTE_OFF        | 62    | Release D4
5        | 318   | VELOCITY_CHANGE | 13    | Set velocity ~52
...
35954    | 1     | END_TOKEN       | 0     | End of piece
```

### Event Type Distribution (Waldstein)
- **NOTE_ON**: 8,576 (23.9%)
- **NOTE_OFF**: 8,576 (23.9%)
- **INSTRUMENT_CHANGE**: 7,181 (20.0%)
- **VELOCITY_CHANGE**: 6,931 (19.3%)
- **TEMPO_CHANGE**: 3,175 (8.8%)
- **TIME_SHIFT**: 1,515 (4.2%)

## 🔤 Vocabulary Usage

- **Total Vocabulary**: 387 tokens
- **Tokens Used**: 236 (61.0% coverage)
- **Token Distribution**:
  - Note events: 64.4%
  - Other events: 25.8%
  - Time events: 8.9%
  - Special tokens: 0.8%

## 💡 Key Insights

### 1. **Rich Musical Complexity**
The classical pieces show significant complexity with thousands of tempo changes (rubato), frequent velocity changes (expressive dynamics), and dense note sequences.

### 2. **Lower Than Expected Polyphony**
Maximum 5 simultaneous notes observed vs expected 8-12 for complex piano music. This suggests:
- MIDI files may have simplified voicing
- Or our test sample didn't include the most complex sections

### 3. **Expressive Performance Data**
High number of tempo changes (3,175 in Waldstein) indicates these are performance MIDIs with human expression, not just notation.

### 4. **Efficient Tokenization**
The 387-token vocabulary captures 61% usage on classical piano, suggesting good coverage while leaving room for other instruments and styles.

## 🎯 Training Implications

### Strengths for Training:
- **High quality classical piano data** with expression
- **Diverse pitch and velocity ranges**
- **Complex temporal patterns** with rubato
- **Clean, well-structured MIDI files**

### Considerations:
- **Limited polyphony** may not capture full piano complexity
- **Classical-heavy dataset** - may need genre diversity
- **Performance-oriented** - includes human timing variations

### Recommendations:
1. Add more contemporary and jazz pieces for diversity
2. Include pieces with higher polyphony (8-12 notes)
3. Consider normalizing extreme tempo variations
4. Add non-piano instruments for broader training

## 📈 Next Steps

With this understanding of the data, we're ready for:
- **Phase 2.3**: Preprocessing pipeline to handle tempo normalization
- **Phase 2.4**: Augmentation to increase diversity
- **Phase 3**: Model architecture optimized for this data profile