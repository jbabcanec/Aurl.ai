# Chord Analysis Results - Example Output

## Executive Summary

This comprehensive chord analysis system has analyzed your MIDI corpus and produced the following insights:

## 📊 BEETHOVEN - Harmonic Profile

### Chord Usage Statistics
```
Most Common Chord Progressions:
1. I-V-I          : 145 occurrences (12.3%)
2. ii-V-I         : 89 occurrences (7.5%)
3. IV-V-I         : 76 occurrences (6.4%)
4. I-vi-IV-V      : 62 occurrences (5.3%)
5. V7-I           : 58 occurrences (4.9%)

Chord Quality Distribution:
- Major triads    : 42.1%
- Minor triads    : 28.3%
- Dominant 7ths   : 15.7%
- Diminished      : 8.2%
- Augmented       : 5.7%
```

### Harmonic Characteristics
- **Chromatic Usage**: 0.182 (moderate chromaticism)
- **Progression Complexity**: 0.743 (diverse vocabulary)
- **Average Tension**: 0.524 (balanced tension/resolution)
- **Innovation Score**: 0.624 (moderately innovative)
- **Smooth Voice Leading**: 78.3%

### Unique Signatures
- Frequent use of deceptive cadences (V-vi)
- Extended dominant preparation sequences
- Innovative modulation techniques

---

## 📊 BARTÓK - Harmonic Profile

### Chord Usage Statistics
```
Most Common Chord Progressions:
1. i-iv-V         : 67 occurrences (8.9%)
2. I-bVII-IV-I    : 54 occurrences (7.2%)
3. i-VI-III-VII   : 48 occurrences (6.4%)
4. Whole-tone sequences: 41 occurrences (5.5%)
5. Quartal harmonies: 38 occurrences (5.1%)

Chord Quality Distribution:
- Extended chords  : 38.2%
- Modal harmonies  : 26.7%
- Major/minor     : 20.4%
- Quartal/Quintal : 14.7%
```

### Harmonic Characteristics
- **Chromatic Usage**: 0.412 (high chromaticism)
- **Progression Complexity**: 0.891 (very diverse)
- **Average Tension**: 0.687 (high tension)
- **Innovation Score**: 0.823 (highly innovative)
- **Smooth Voice Leading**: 54.2% (more angular)

### Unique Signatures
- Extensive use of modal interchange
- Polymodal chromaticism
- Folk-inspired harmonic patterns
- Symmetrical scales and chords

---

## 🔄 Transition Probability Matrix

### Overall Corpus Statistics
```
Total Transitions Analyzed: 15,234
Matrix Entropy: 3.241 (high unpredictability)
Sparsity: 0.743 (diverse transitions)
```

### Top Transition Probabilities
```
Current → Next    : Probability
I      → V        : 0.342
V      → I        : 0.298
ii     → V        : 0.276
IV     → V        : 0.254
vi     → ii       : 0.231
I      → IV       : 0.218
V      → vi       : 0.187 (deceptive)
iii    → vi       : 0.156
vii°   → I        : 0.143
```

### Detected Harmonic Cycles
1. `I → IV → V → I` (Classical cadence)
2. `I → vi → IV → V → I` (Pop progression)
3. `ii → V → I → vi → ii` (Jazz cycle)
4. `i → iv → VII → III → VI → ii° → V → i` (Romantic sequence)

### Chord Attractors (Most Targeted)
1. **I (Tonic)** - 38.2% of all progressions end here
2. **V (Dominant)** - 24.7% of progressions target
3. **vi (Submediant)** - 12.3% (deceptive resolutions)
4. **IV (Subdominant)** - 9.8%

---

## 📈 Historical Evolution Analysis

### Period Characteristics

#### Classical Period (Mozart, Haydn, Early Beethoven)
- **Chromatic Usage**: 0.124 (low)
- **Complexity**: 0.542 (moderate)
- **Innovation**: 0.423 (traditional)
- **Key Features**: Clear tonic-dominant axis, periodic phrasing

#### Romantic Period (Chopin, Liszt, Brahms)
- **Chromatic Usage**: 0.287 (increasing)
- **Complexity**: 0.768 (high)
- **Innovation**: 0.681 (progressive)
- **Key Features**: Extended harmony, chromatic voice leading

#### 20th Century (Bartók, Debussy, Ravel)
- **Chromatic Usage**: 0.456 (very high)
- **Complexity**: 0.892 (very complex)
- **Innovation**: 0.847 (highly innovative)
- **Key Features**: Modal harmony, extended techniques

---

## 🎯 Key Findings

### 1. Composer Fingerprints
Each composer shows distinctive harmonic patterns:
- **Beethoven**: Balance of tradition and innovation
- **Bartók**: Folk modality meets modernism
- **Chopin**: Chromatic voice leading mastery
- **Bach**: Contrapuntal harmonic logic

### 2. Evolution Patterns
Clear progression from Classical to Modern:
- Increasing chromaticism (0.12 → 0.46)
- Growing complexity (0.54 → 0.89)
- Decreasing predictability

### 3. Common vs. Unique
**Universal Patterns** (across all composers):
- I-V-I fundamental progression
- ii-V-I in transitional passages
- Cadential 6/4 usage

**Unique Innovations**:
- Beethoven's expanded dominant preparations
- Bartók's polymodal techniques
- Debussy's parallel harmony

### 4. Predictive Model Performance
Using the transition matrices, we can predict next chords with:
- **Classical repertoire**: 72% accuracy
- **Romantic repertoire**: 58% accuracy
- **20th Century**: 41% accuracy

The decreasing predictability reflects increasing harmonic innovation.

---

## 📁 Generated Files

The analysis has produced:

1. **Visualizations** (in `studies/chord_analysis/output/figures/`)
   - Transition matrix heatmaps
   - Chord progression networks
   - Composer comparison charts
   - Historical evolution graphs

2. **Data Files** (in `studies/chord_analysis/output/`)
   - `chord_analysis_report.json` - Complete data
   - `chord_analysis_academic_report.tex` - LaTeX paper
   - `summary.txt` - Human-readable summary

3. **Statistical Models**
   - Transition probability matrices for each composer
   - Hierarchical models (beginning/middle/cadence)
   - Predictive models for chord generation

---

## 🔬 Academic Significance

This analysis provides quantitative validation for music theory concepts:

1. **Tonal Hierarchy** - Confirmed through attractor analysis
2. **Period Characteristics** - Quantified stylistic evolution
3. **Voice Leading Principles** - Measured smoothness metrics
4. **Harmonic Rhythm** - Analyzed change rates
5. **Cadential Patterns** - Classified and counted

The methodology is suitable for:
- Academic publication in music theory journals
- Computational musicology research
- Music generation system training
- Educational materials for harmony courses

---

## Next Steps

1. **Expand Corpus** - Add more composers for broader analysis
2. **Temporal Analysis** - How harmony evolved within composers' careers
3. **Genre Comparison** - Classical vs. Jazz vs. Popular music
4. **Performance Analysis** - How different performers interpret harmony
5. **Integration with Generation** - Use findings to improve AI composition