import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import beta, norm
import pickle
import json

def calculate_tolerance_limits(n):
    """
    Calculate outer and inner tolerance limits (oTL, iTL) based on sample size.
    Uses the exact R tolLimits function logic.
    
    Args:
        n (int): Sample size
        
    Returns:
        dict: Dictionary containing oTL, iTL observation numbers and their probabilities
    """
    q = 0.05
    r1 = r2 = 0
    
    # Calculate outer tolerance limit (oTL)
    # Start with r1 = 1 to avoid beta parameters <= 0
    r1 = 1
    while r1 < n:
        shape1 = r1
        shape2 = n - r1 - 1
        
        if shape2 <= 0:
            break
            
        p1 = beta.cdf(q, a=shape1, b=shape2)
        
        if p1 < 0.95:
            break
        r1 += 1
    
    # Step back one as per R logic
    r1 = r1 - 1
    
    # Calculate final p1
    if r1 >= 1:
        shape1 = r1
        shape2 = n - r1 - 1
        if shape2 > 0:
            p1 = beta.cdf(q, a=shape1, b=shape2)
        else:
            p1 = 0.0
            r1 = 0  # Set to "not defined"
    else:
        r1 = 0
        p1 = 0.0
    
    # Calculate inner tolerance limit (iTL)
    r2 = 1  # Start from 1 to avoid shape2=0
    while r2 < n:
        shape1 = n - r2 - 1
        shape2 = r2
        
        if shape1 <= 0:
            break
            
        p2 = beta.cdf(1-q, a=shape1, b=shape2)
        
        if p2 > 0.95:
            break
        r2 += 1
    
    # Calculate final p2
    shape1 = n - r2 - 1
    shape2 = r2
    if shape1 > 0 and shape2 > 0:
        p2 = beta.cdf(1-q, a=shape1, b=shape2)
    else:
        p2 = 0.0
    
    # Set oTL
    oTL = 'not defined' if r1 == 0 else r1
    iTL = r2
    
    return {
        'oTL_obs': oTL,
        'iTL_obs': iTL,
        'p_oTL': round(p1, 5) if oTL != 'not defined' else 0.0,
        'p_iTL': round(p2, 5)
    }

def calculate_es_thresholds(n, oTL_obs):
    """
    Calculate Equivalent Scores (ES) thresholds based on sample size and outer tolerance limit.
    Uses the exact R logic for ES calculation.
    
    Args:
        n (int): Sample size
        oTL_obs (int): Outer tolerance limit observation number
        
    Returns:
        dict: Dictionary containing ES1, ES2, ES3 observation thresholds
    """
    if oTL_obs == 'not defined':
        raise ValueError("Cannot calculate ES thresholds when oTL is not defined")
    
    # Initial calculations
    cd1 = oTL_obs / n
    
    # Check if cd1 is valid for inverse normal
    if cd1 <= 0 or cd1 >= 1:
        raise ValueError(f"Invalid probability value cd1 = {cd1}. Must be between 0 and 1.")
    
    z1 = norm.ppf(cd1)
    
    # Check if z1 is valid
    if np.isnan(z1) or np.isinf(z1):
        raise ValueError(f"Invalid z-score calculated: z1 = {z1}")
    
    z1_3 = z1 / 3
    z1_2 = z1_3 * 2
    
    # Calculate ES1
    cd2 = norm.cdf(z1_2)
    a = (cd1 - cd2) * n
    
    # Check for NaN values before rounding
    if np.isnan(a) or np.isinf(a):
        raise ValueError(f"Invalid value calculated: a = {a}")
    
    a_r = int(round(a))
    ES1_obs = -a_r + oTL_obs
    
    # Calculate ES2
    cd3 = norm.cdf(z1_3)
    b = (cd3 - cd2) * n
    
    if np.isnan(b) or np.isinf(b):
        raise ValueError(f"Invalid value calculated: b = {b}")
    
    b_r = int(round(b))
    ES2_obs = ES1_obs + b_r
    
    # Calculate ES3
    cd4 = norm.cdf(0)  # This equals 0.5
    c = (cd4 - cd3) * n
    
    if np.isnan(c) or np.isinf(c):
        raise ValueError(f"Invalid value calculated: c = {c}")
    
    c_r = int(round(c))
    ES3_obs = ES2_obs + c_r
    
    return {
        'ES1_obs': ES1_obs,
        'ES2_obs': ES2_obs,
        'ES3_obs': ES3_obs
    }

def adjust_scores(data, score_var):
    """
    Fit OLS regression model to adjust scores for demographics
    """
    # Prepare features
    X = data[['age', 'sex_encoded', 'education']]
    y = data[score_var]
    
    # Add constant for intercept
    X_with_const = sm.add_constant(X)
    
    # Fit OLS model
    model = sm.OLS(y, X_with_const).fit()
    
    # Get predictions
    preds = model.predict(X_with_const)
    
    # Adjusted score = raw score - (predicted - mean predicted)
    adj_scores = y - (preds - np.mean(preds))
    
    return adj_scores, model

def compute_es_thresholds_from_scores(adjusted_scores):
    """
    Compute oTL, iTL and thresholds for ES categories using the R methodology
    """
    sorted_scores = np.sort(adjusted_scores)
    N = len(sorted_scores)
    
    # Calculate tolerance limits using the R methodology
    tol_limits = calculate_tolerance_limits(N)
    
    if tol_limits['oTL_obs'] == 'not defined':
        raise ValueError(f"Sample size {N} too small to define outer tolerance limit")
    
    # Get actual score values at tolerance limit positions
    oTL_score = sorted_scores[tol_limits['oTL_obs'] - 1]  # -1 for 0-based indexing
    iTL_score = sorted_scores[tol_limits['iTL_obs'] - 1]
    
    # Calculate ES thresholds using R methodology
    try:
        es_limits = calculate_es_thresholds(N, tol_limits['oTL_obs'])
        
        # Convert observation numbers to actual scores
        ES1_score = sorted_scores[es_limits['ES1_obs'] - 1] if es_limits['ES1_obs'] > 0 else sorted_scores[0]
        ES2_score = sorted_scores[es_limits['ES2_obs'] - 1] if es_limits['ES2_obs'] > 0 else sorted_scores[0]
        ES3_score = sorted_scores[es_limits['ES3_obs'] - 1] if es_limits['ES3_obs'] > 0 else sorted_scores[0]
        
        thresholds = {
            'oTL': oTL_score,
            'iTL': iTL_score,
            'threshold_1': ES1_score,  # ES0/ES1 boundary
            'threshold_2': ES2_score,  # ES1/ES2 boundary  
            'threshold_3': ES3_score,  # ES2/ES3 boundary
            'median': np.median(sorted_scores),
            'oTL_obs': tol_limits['oTL_obs'],
            'iTL_obs': tol_limits['iTL_obs'],
            'ES1_obs': es_limits['ES1_obs'],
            'ES2_obs': es_limits['ES2_obs'],
            'ES3_obs': es_limits['ES3_obs'],
            'p_oTL': tol_limits['p_oTL'],
            'p_iTL': tol_limits['p_iTL']
        }
        
    except ValueError as e:
        # Fallback to simple percentile method if R method fails
        print(f"Warning: R method failed ({e}), using percentile method")
        
        # Simple percentile-based thresholds as fallback
        oTL_score = np.percentile(sorted_scores, 5)
        iTL_score = np.percentile(sorted_scores, 95)
        median = np.median(sorted_scores)
        
        threshold_1 = oTL_score + (median - oTL_score) / 3
        threshold_2 = oTL_score + 2 * (median - oTL_score) / 3
        threshold_3 = median
        
        thresholds = {
            'oTL': oTL_score,
            'iTL': iTL_score,
            'threshold_1': threshold_1,
            'threshold_2': threshold_2,
            'threshold_3': threshold_3,
            'median': median,
            'method': 'percentile_fallback'
        }
    
    return thresholds

def compute_iverson_classification(percentiles):
    """
    Apply Iverson collective classification rules based on percentiles
    
    Args:
        percentiles: list of 4 percentile values (0-100)
    
    Returns:
        classification: string describing the overall cognitive status
    """
    # Count scores below various thresholds
    below_25th = sum(p <= 25 for p in percentiles)
    below_16th = sum(p <= 16 for p in percentiles)
    below_10th = sum(p <= 10 for p in percentiles)
    below_5th = sum(p <= 5 for p in percentiles)
    below_2nd = sum(p <= 2 for p in percentiles)
    
    # Apply Iverson rules (in order of severity - most severe first)
    if below_5th >= 3 or below_2nd >= 2:
        return "Extremely Low"
    elif below_10th >= 3 or below_5th >= 2 or any(p <= 2 for p in percentiles):
        return "Unusually Low"
    elif below_16th >= 3 or below_10th >= 2 or any(p <= 5 for p in percentiles):
        return "Well Below Average"
    elif below_25th >= 3 or below_16th >= 2 or any(p <= 10 for p in percentiles):
        return "Below Average"
    else:
        return "Broadly Normal"

def get_iverson_interpretation(classification):
    """
    Provide clinical interpretation and color coding for Iverson classifications
    """
    interpretations = {
        "Broadly Normal": {
            "color": "#388e3c", 
            "description": "Cognitive performance within normal limits across domains",
            "clinical_note": "No significant cognitive concerns identified"
        },
        "Below Average": {
            "color": "#fbc02d", 
            "description": "Mild cognitive difficulties across multiple domains or isolated weakness",
            "clinical_note": "Consider monitoring and supportive interventions"
        },
        "Well Below Average": {
            "color": "#f57c00", 
            "description": "Moderate cognitive difficulties requiring attention",
            "clinical_note": "Comprehensive assessment and intervention recommended"
        },
        "Unusually Low": {
            "color": "#e65100", 
            "description": "Significant cognitive impairment across domains",
            "clinical_note": "Immediate clinical attention and specialized assessment needed"
        },
        "Extremely Low": {
            "color": "#d32f2f", 
            "description": "Severe cognitive impairment requiring urgent intervention",
            "clinical_note": "Urgent referral for specialized neuropsychological evaluation"
        }
    }
    return interpretations[classification]

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv("impact_phi_reading.csv")
    
    # Transform reaction speed to reaction efficiency (higher = better)
    # This ensures all cognitive measures follow the same direction
    df['reaction_efficiency'] = 1 / df['reaction_speed']
    
    # Encode sex variable
    le_sex = LabelEncoder()
    df['sex_encoded'] = le_sex.fit_transform(df['sex'])
    
    # Cognitive score variables - use reaction_efficiency instead of reaction_speed
    score_vars = ["visual_memory", "verbal_memory", "visual_motor_speed", "reaction_efficiency"]
    
    # Store results
    models = {}
    thresholds = {}
    sample_stats = {}
    
    print("Fitting models and computing thresholds...")
    
    for score_var in score_vars:
        print(f"Processing {score_var}...")
        
        # Adjust scores
        adj_scores, model = adjust_scores(df, score_var)
        
        # Compute ES thresholds using R methodology
        score_thresholds = compute_es_thresholds_from_scores(adj_scores)
        
        # Store results - but keep original naming for reaction_speed in output
        output_var = 'reaction_speed' if score_var == 'reaction_efficiency' else score_var
        models[output_var] = model
        thresholds[output_var] = score_thresholds
        
        # Compute sample statistics
        if score_var == 'reaction_efficiency':
            # For reaction speed, store stats about the original reaction times
            sample_stats['reaction_speed'] = {
                'n_samples': len(df),
                'raw_mean': float(df['reaction_speed'].mean()),
                'raw_std': float(df['reaction_speed'].std()),
                'adj_mean': float(np.mean(adj_scores)),  # This is for efficiency
                'adj_std': float(np.std(adj_scores)),
                'min_score': float(df['reaction_speed'].min()),
                'max_score': float(df['reaction_speed'].max()),
                'mse_resid': float(model.mse_resid)  # For Iverson method
            }
        else:
            sample_stats[score_var] = {
                'n_samples': len(df),
                'raw_mean': float(df[score_var].mean()),
                'raw_std': float(df[score_var].std()),
                'adj_mean': float(np.mean(adj_scores)),
                'adj_std': float(np.std(adj_scores)),
                'min_score': float(df[score_var].min()),
                'max_score': float(df[score_var].max()),
                'mse_resid': float(model.mse_resid)  # For Iverson method
            }
    
    # Save models and model summaries
    print("Saving models...")
    with open('cognitive_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    # Save model summaries as text for review
    model_summaries = {}
    for var, model in models.items():
        model_summaries[var] = {
            'summary': str(model.summary()),
            'params': model.params.to_dict(),
            'pvalues': model.pvalues.to_dict(),
            'rsquared': float(model.rsquared),
            'rsquared_adj': float(model.rsquared_adj),
            'fvalue': float(model.fvalue),
            'f_pvalue': float(model.f_pvalue)
        }
    
    with open('model_summaries.json', 'w') as f:
        json.dump(model_summaries, f, indent=2)
    
    # Save thresholds and sample stats
    with open('model_parameters.json', 'w') as f:
        json.dump({
            'thresholds': thresholds,
            'sample_stats': sample_stats,
            'sex_classes': le_sex.classes_.tolist(),
            'demographic_ranges': {
                'age': {'min': float(df['age'].min()), 'max': float(df['age'].max())},
                'education': {'min': float(df['education'].min()), 'max': float(df['education'].max())}
            }
        }, f, indent=2)
    
    print("Model fitting completed!")
    print(f"Processed {len(df)} samples across {len(score_vars)} cognitive domains")
    print("Files saved: cognitive_models.pkl, model_parameters.json, model_summaries.json")
    
    # Display summary
    print("\n" + "="*60)
    print("SAMPLE SUMMARY")
    print("="*60)
    print(f"Total Sample Size: {len(df)}")
    print(f"Age Range: {df['age'].min():.1f} - {df['age'].max():.1f} years")
    print(f"Education Range: {df['education'].min():.1f} - {df['education'].max():.1f} years")
    print(f"Sex Distribution: {df['sex'].value_counts().to_dict()}")
    
    # Display tolerance limits information
    print("\n" + "="*60)
    print("TOLERANCE LIMITS ANALYSIS")
    print("="*60)
    display_vars = ["visual_memory", "verbal_memory", "visual_motor_speed", "reaction_speed"]
    for var in display_vars:
        print(f"\n{var.replace('_', ' ').title()}:")
        thresh = thresholds[var]
        if 'oTL_obs' in thresh:
            print(f"  Sample Size: {sample_stats[var]['n_samples']}")
            print(f"  Outer Tolerance Limit: observation {thresh['oTL_obs']} (safety level {thresh['p_oTL']*100:.1f}%)")
            print(f"  Inner Tolerance Limit: observation {thresh['iTL_obs']} (safety level {thresh['p_iTL']*100:.1f}%)")
            print(f"  ES Thresholds: obs {thresh['ES1_obs']}, {thresh['ES2_obs']}, {thresh['ES3_obs']}")
        else:
            print(f"  Used percentile fallback method")
    
    print("\n" + "="*60)
    print("MODEL STATISTICS")
    print("="*60)
    display_vars = ["visual_memory", "verbal_memory", "visual_motor_speed", "reaction_speed"]
    for var in display_vars:
        model = models[var]
        print(f"\n{var.replace('_', ' ').title()}:")
        print(f"  R-squared: {model.rsquared:.3f}")
        print(f"  Adj. R-squared: {model.rsquared_adj:.3f}")
        print(f"  F-statistic: {model.fvalue:.2f} (p={model.f_pvalue:.3f})")
        print(f"  Significant predictors (p<0.05):")
        for param, pval in model.pvalues.items():
            if pval < 0.05:
                coef = model.params[param]
                print(f"    {param}: β={coef:.3f} (p={pval:.3f})")
    
    print("\n" + "="*60)
    print("TOLERANCE LIMITS BY DOMAIN")
    print("="*60)
    display_vars = ["visual_memory", "verbal_memory", "visual_motor_speed", "reaction_speed"]
    for score_var in display_vars:
        print(f"\n{score_var.replace('_', ' ').title()}:")
        if score_var == 'reaction_speed':
            print(f"  Note: Thresholds based on reaction efficiency (1/time)")
        print(f"  oTL (worst 5%): {thresholds[score_var]['oTL']:.3f}")
        print(f"  ES1 threshold: {thresholds[score_var]['threshold_1']:.3f}")
        print(f"  ES2 threshold: {thresholds[score_var]['threshold_2']:.3f}")
        print(f"  ES3 threshold: {thresholds[score_var]['threshold_3']:.3f}")
        print(f"  Median: {thresholds[score_var]['median']:.3f}")
        print(f"  iTL (best 5%): {thresholds[score_var]['iTL']:.3f}")

if __name__ == "__main__":
    main()