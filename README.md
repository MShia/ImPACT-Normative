# Cognitive Assessment Dashboard

A clinical assessment tool for ImPACT based Concussion evaluation using two validated methodologies: Individual Equivalent Scores (ES) and Iverson Collective Classification.

## Features

- **Dual Assessment Methods**: Choose between Individual ES Method (0-4 scoring) or Iverson Collective Classification
- **Demographic Adjustment**: Automatic score adjustment for age, sex, and education
- **Statistical Tolerance Limits**: Uses Non Parametric Tolerance limits and ES methodology for precise threshold calculations
- **Professional Reporting**: Generate comprehensive Word documents with detailed analysis
- **Interactive Visualizations**: Gauge charts and color-coded displays for clear result interpretation
- **Clinical Decision Support**: Supports the Clinicians to make informed decisions as per the exisiting statistical methodologies and Normative data from the relevant cohort for Jockeys 

## Assessment Methods

### Individual ES Method
- Provides domain-specific scores on a 0-4 scale (Impaired, Borderline, low normal, Normal, High Normal)
- Based on statistically computed tolerance limits using beta distributions
- Each cognitive domain evaluated independently
- Scores adjusted for demographic factors (age, sex, education)

### Iverson Collective Classification
- Holistic assessment considering patterns across all domains
- Classifications: Broadly Normal, Below Average, Well Below Average, Unusually Low, Extremely Low
- Based on percentile thresholds and collective cognitive profile
- Follows established neuropsychological criteria

## Cognitive Domains

The dashboard assesses four key cognitive domains:

1. **Visual Memory**: Spatial and visual information retention
2. **Verbal Memory**: Language-based information recall
3. **Visual-Motor Speed**: Processing speed and coordination
4. **Reaction Speed**: Response time efficiency (converted to efficiency scores)

## Installation

### Prerequisites

```bash
pip install streamlit pandas numpy scipy statsmodels scikit-learn plotly python-docx
```

### Required Files

Ensure these files are in your project directory:
- `cognitive_models.pkl` - Trained regression models
- `model_parameters.json` - Statistical thresholds and sample statistics
- `model_summaries.json` - Model performance metrics

### Generate Model Files

Run the model fitting pipeline first:

```python
python fit_models.py
```

This will create the required model files from your normative dataset.

## Usage

### Start the Dashboard

```bash
streamlit run dashboard.py
```

### Using the Interface

1. **Select Assessment Method**: Choose between Individual ES or Iverson Classification
2. **Enter Demographics**: Age, sex, and education (with automatic validation)
3. **Input Test Scores**: Raw cognitive test results
4. **Generate Assessment**: Click "COMPUTE ASSESSMENT" for results
5. **Download Reports**: Generate professional Word reports

### Input Validation

- **Age Range**: Limited by normative sample boundaries
- **Education Validation**: Automatically constrains education years based on age (assuming school starts at age 5)
- **Score Ranges**: Validated against expected test score distributions

## Statistical Methodology

### Tolerance Limits Calculation

The dashboard uses existing statistical methods for threshold determination: using Statesmodels and other libraries, see the requirement.txt libraries for more information

```python
# Outer and Inner Tolerance Limits using beta distributions
oTL = beta_distribution_analysis(sample_size, confidence_level=0.95)
iTL = beta_distribution_analysis(sample_size, confidence_level=0.95)

# ES Thresholds using normal distribution transformations  
ES_thresholds = normal_distribution_partitioning(oTL, sample_size)
```

### Demographic Adjustment

Scores are adjusted using OLS regression:

```python
adjusted_score = raw_score - (predicted_score - mean_predicted_score)
```

Where predicted scores account for:
- Age effects (linear, quadratic, cubic, logarithmic transformations)
- Sex differences
- Education level impact

## Output Interpretation

### ES Scores (0-4 Scale)

- **ES 0**: Impaired 
- **ES 1**: Borderline
- **ES 2**: Low Normal 
- **ES 3**: Normal 
- **ES 4**: High Normal

### Iverson Classifications

- **Broadly Normal**: No significant cognitive concerns
- **Below Average**: Mild difficulties, monitoring recommended
- **Well Below Average**: Moderate difficulties, comprehensive assessment needed
- **Unusually Low**: Significant impairment, immediate attention required
- **Extremely Low**: Severe impairment, urgent specialized evaluation

## File Structure

```
cognitive-assessment/
├── impact_dash_streamlit.py                 # Main Streamlit application
├── fit_models.py               # Model training pipeline
├── cognitive_models.pkl        # Trained regression models
├── model_parameters.json       # Statistical parameters
├── model_summaries.json        # Model performance metrics
├── your_raw_file.csv           # Normative dataset - Use your own file to generate model
└── README.md                   # This file
```

## Technical Specifications

### Dependencies

- **Streamlit**: Web interface framework
- **Pandas/NumPy**: Data manipulation and numerical computing
- **SciPy**: Statistical distributions and calculations
- **Statsmodels**: Regression modeling
- **Plotly**: Interactive visualizations
- **python-docx**: Word document generation

### Performance

- **Model Fitting**: Processes datasets up to 10,000+ participants
- **Real-time Assessment**: Sub-second response times
- **Memory Usage**: ~50MB for typical normative samples
- **Supported Browsers**: Chrome, Firefox, Safari, Edge

## Validation and References

This implementation follows established methodologies:

1. **Aiello EN, Depaoli EG** (2022). Norms and standardizations in neuropsychology via equivalent scores: software solutions and practical guides. *Neurol Sci* 43(2):961-966.

2. **Iverson GL, Webbe FM** (2011). Evidence-based neuropsychological assessment in sport-related concussion. *The handbook of sport neuropsychology*, 131-153.

## Clinical Use Guidelines

### Intended Use
- Cognitive screening and assessment
- Research applications
- Educational evaluations
- Clinical decision support

### Limitations
- Requires qualified professional interpretation
- Not a standalone diagnostic tool
- Must consider clinical context and patient history
- Normative sample representativeness affects accuracy

### Best Practices
- Always interpret results in clinical context
- Consider test conditions and patient factors
- Use alongside other assessment tools
- Follow up abnormal results with comprehensive evaluation

## Contributing

When contributing to this project:

1. Ensure statistical methods maintain scientific rigor
2. Validate against established neuropsychological standards
3. Test with diverse demographic samples
4. Document any methodological changes
5. Maintain clinical safety and ethical guidelines

## License

This software is intended for clinical and research use. Ensure compliance with local regulations regarding medical software and patient data handling.

## Support

For technical issues or methodological questions:
- Check model file generation first
- Verify input data format and ranges
- Review statistical assumptions and sample characteristics
- Consider normative sample appropriateness for your population

## Disclaimer

This tool provides automated cognitive assessment support but requires professional interpretation. Results should be integrated with clinical judgment, patient history, and other relevant information. Not intended as a standalone diagnostic instrument.