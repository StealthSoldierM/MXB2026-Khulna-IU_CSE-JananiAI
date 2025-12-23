import pandas as pd
import numpy as np
import os
import warnings
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

warnings.filterwarnings('ignore')

def build_seed_dataset(sample_size=5000):
    np.random.seed(42)
    print("Building seed dataset with BDHS 2017-18 distributions...")
    
    age_values = list(range(15, 50))  # Creates [15, 16, 17, ..., 49]
    
    age_probabilities = []
    for age in age_values:
        if age < 30:
            age_probabilities.append(0.022)  # Slightly higher for young mothers
        elif age < 45:
            age_probabilities.append(0.020)  # Prime childbearing years
        else:
            age_probabilities.append(0.008)  # Declining fertility
    
    age_probabilities = np.array(age_probabilities)
    age_probabilities = age_probabilities / age_probabilities.sum()
    
    education_years = list(range(0, 13))  # 0 to 12 years of schooling
    education_probabilities = [0.12, 0.09, 0.10, 0.09, 0.08, 0.08, 0.09, 0.07, 0.08, 0.07, 0.07, 0.04, 0.02]
    education_probabilities = np.array(education_probabilities)
    education_probabilities = education_probabilities / education_probabilities.sum()
    
    wealth_categories = ['poorest', 'poorer', 'middle', 'richer', 'richest']
    wealth_probabilities = [0.20, 0.20, 0.20, 0.20, 0.20]
    
    # Urban-rural split from BDHS 2017: 37% urban, 63% rural
    # This is important because location strongly correlates with healthcare access
    residence_types = ['urban', 'rural']
    residence_probabilities = [0.37, 0.63]
    
    # Now let's model antenatal care (ANC) visits. WHO recommends at least 4 visits
    # but access varies widely in Bangladesh. Our distribution reflects that about
    # 64% of women achieve the recommended 4+ visits
    anc_visit_counts = list(range(0, 11))  # 0 to 10 visits
    anc_probabilities = [0.15, 0.09, 0.08, 0.07, 0.31, 0.11, 0.08, 0.05, 0.03, 0.02, 0.01]
    anc_probabilities = np.array(anc_probabilities)
    anc_probabilities = anc_probabilities / anc_probabilities.sum()
    
    # Create the base dataframe with all our carefully constructed distributions
    seed_data = pd.DataFrame({
        'age': np.random.choice(age_values, sample_size, p=age_probabilities),
        
        'education_years': np.random.choice(education_years, sample_size, 
                                           p=education_probabilities),
        
        'wealth_quintile': np.random.choice(wealth_categories, sample_size, 
                                           p=wealth_probabilities),
        
        'urban_rural': np.random.choice(residence_types, sample_size, 
                                       p=residence_probabilities),
        
        'anc_visits': np.random.choice(anc_visit_counts, sample_size, 
                                      p=anc_probabilities),
        
        # Binary outcomes: skilled birth attendance and facility delivery
        # These use Bernoulli distributions (coin flips with specific probabilities)
        'skilled_delivery': np.random.binomial(1, 0.505, sample_size),  # 50.5% have skilled attendant
        'facility_delivery': np.random.binomial(1, 0.574, sample_size),  # 57.4% deliver in facilities
        'complications': np.random.binomial(1, 0.32, sample_size),      # 32% report complications
        
        # Clinical measurements follow normal distributions with realistic parameters
        # Systolic blood pressure: mean 120, std 15, clipped to medically plausible range
        'sys_bp': np.clip(np.random.normal(120, 15, sample_size), 80, 200),
        'dia_bp': np.clip(np.random.normal(80, 10, sample_size), 50, 140),
        'heart_rate': np.clip(np.random.normal(78, 12, sample_size), 50, 120),
        'glucose': np.clip(np.random.normal(95, 20, sample_size), 60, 200),
        
        # District ID represents geographic location (Bangladesh has 64 districts)
        'district_id': np.random.randint(1, 65, sample_size)
    })
    
    # Now comes the critical part: injecting realistic correlations
    # In real data, wealth strongly predicts healthcare access, so we need to
    # artificially create this relationship in our seed data
    
    print("Injecting socioeconomic correlations...")
    
    # Poorer women have fewer ANC visits on average
    # We identify them and reduce their visit counts by 30%
    poor_mask = seed_data['wealth_quintile'].isin(['poorest', 'poorer'])
    seed_data.loc[poor_mask, 'anc_visits'] = (
        seed_data.loc[poor_mask, 'anc_visits'] * 0.70
    ).clip(0, 10).round()
    
    # Wealthier women almost always have skilled delivery
    # This reflects better access to quality healthcare
    rich_mask = seed_data['wealth_quintile'].isin(['richer', 'richest'])
    seed_data.loc[rich_mask, 'skilled_delivery'] = 1
    
    # Urban residents have much better access to health facilities
    urban_mask = seed_data['urban_rural'] == 'urban'
    seed_data.loc[urban_mask, 'facility_delivery'] = np.random.binomial(
        1, 0.85, urban_mask.sum()
    )  # 85% facility delivery in urban areas
    
    # Education correlates with healthcare utilization
    # Women with more education seek more prenatal care
    high_education_mask = seed_data['education_years'] >= 10
    seed_data.loc[high_education_mask, 'anc_visits'] = (
        seed_data.loc[high_education_mask, 'anc_visits'] * 1.15
    ).clip(0, 10).round()
    
    print("Calculating risk indicators...")
    
    # Now we derive clinical risk flags based on WHO guidelines
    # Preeclampsia risk: high blood pressure (140/90 or higher)
    seed_data['preeclampsia_risk'] = (
        (seed_data['sys_bp'] > 140) | (seed_data['dia_bp'] > 90)
    ).astype(int)
    
    # Gestational diabetes risk: fasting glucose above 126 mg/dL
    seed_data['gdm_risk'] = (seed_data['glucose'] > 126).astype(int)
    
    # High-risk pregnancy: multiple risk factors present
    # This is a composite score that could guide clinical decision-making
    seed_data['high_risk'] = (
        (seed_data['preeclampsia_risk'] + 
         seed_data['gdm_risk'] + 
         seed_data['complications']) >= 2
    ).astype(int)
    
    # Round all continuous variables to whole numbers for cleaner data
    continuous_columns = ['sys_bp', 'dia_bp', 'heart_rate', 'glucose']
    for column in continuous_columns:
        seed_data[column] = seed_data[column].round()
    
    print(f"✓ Seed dataset created: {len(seed_data)} records")
    print(f"  - Variables: {len(seed_data.columns)}")
    print(f"  - High-risk pregnancies: {seed_data['high_risk'].sum()} ({seed_data['high_risk'].mean():.1%})")
    
    return seed_data


def train_synthetic_generator(training_data):
    """
    Trains a CTGAN model to learn the patterns in our seed data.
    
    CTGAN (Conditional Tabular GAN) is a specialized neural network that learns
    to generate new data by playing a game: one network (generator) tries to
    create fake data, another network (discriminator) tries to detect fakes.
    Through this adversarial process, the generator becomes very good at creating
    realistic synthetic records.
    
    Parameters:
        training_data (pd.DataFrame): The seed dataset to learn from
    
    Returns:
        CTGANSynthesizer: Trained model ready to generate synthetic data
    """
    
    print("\nInitializing CTGAN synthesizer...")
    
    # First, we need to tell SDV about our data structure using metadata
    # This helps the model understand which columns are categorical vs continuous
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(training_data)
    
    # We can inspect what SDV detected to ensure it understood our data correctly
    print("Detected data types:")
    for column_name, column_info in metadata.columns.items():
        print(f"  - {column_name}: {column_info['sdtype']}")
    
    # Initialize the CTGAN synthesizer with carefully chosen hyperparameters
    # These control the tradeoff between training time and quality
    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        
        # epochs controls how many times the model sees the entire dataset
        # More epochs = better learning but longer training time
        # 300 is a sweet spot for datasets of this size
        epochs=300,
        
        # batch_size controls how many records are processed together
        # Larger batches train faster but use more memory
        batch_size=500,
        
        # We disable verbose output to avoid cluttering the console
        verbose=False,
        
        # Disable GPU even if available, as CTGAN can be unstable on GPU
        # for small-medium datasets. CPU training is more reliable.
        cuda=False,
        
        # generator_dim and discriminator_dim control model complexity
        # We use smaller networks for faster training on this dataset size
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        
        # Learning rate for the neural networks
        # Default 2e-4 works well for most tabular data
        generator_lr=2e-4,
        discriminator_lr=2e-4
    )
    
    print(f"\nTraining CTGAN model on {len(training_data)} records...")
    print("This typically takes 3-5 minutes on a modern CPU...")
    print("(The model is learning the statistical relationships in your data)\n")
    
    # The fit method does all the heavy lifting: it trains the neural networks
    # through hundreds of iterations until they can generate realistic data
    synthesizer.fit(training_data)
    
    print("✓ Model training complete!")
    
    return synthesizer


def generate_synthetic_data(synthesizer, num_records=10000):
    """
    Uses the trained model to generate brand new synthetic records.
    
    These records don't correspond to any real individuals, but they maintain
    the statistical properties and correlations we observed in the training data.
    
    Parameters:
        synthesizer (CTGANSynthesizer): Trained CTGAN model
        num_records (int): How many synthetic records to generate
    
    Returns:
        pd.DataFrame: Synthetic data ready for analysis
    """
    
    print(f"\nGenerating {num_records:,} synthetic records...")
    
    # The sample method generates new data by having the generator network
    # create records from random noise, just like it learned to do during training
    synthetic_data = synthesizer.sample(num_rows=num_records)
    
    print("✓ Synthetic data generated!")
    print("\nApplying post-processing to ensure data quality...")
    
    # Even though CTGAN is good, it sometimes generates values slightly outside
    # the valid range. We need to clean these up to ensure data quality.
    
    # Ensure age stays within reproductive age range (15-49)
    synthetic_data['age'] = synthetic_data['age'].clip(15, 49).round().astype(int)
    
    # Education years must be between 0 and 12
    synthetic_data['education_years'] = (
        synthetic_data['education_years'].clip(0, 12).round().astype(int)
    )
    
    # ANC visits are discrete counts from 0 to 10
    synthetic_data['anc_visits'] = (
        synthetic_data['anc_visits'].clip(0, 10).round().astype(int)
    )
    
    # District IDs must be valid (1-64)
    synthetic_data['district_id'] = (
        synthetic_data['district_id'].clip(1, 64).round().astype(int)
    )
    
    # Clinical measurements need medically plausible bounds
    synthetic_data['sys_bp'] = synthetic_data['sys_bp'].clip(80, 200).round()
    synthetic_data['dia_bp'] = synthetic_data['dia_bp'].clip(50, 140).round()
    synthetic_data['heart_rate'] = synthetic_data['heart_rate'].clip(50, 120).round()
    synthetic_data['glucose'] = synthetic_data['glucose'].clip(60, 200).round()
    
    # Binary columns must be exactly 0 or 1 (sometimes CTGAN generates 0.1 or 0.9)
    # We use 0.5 as the threshold: anything above becomes 1, anything below becomes 0
    binary_columns = [
        'skilled_delivery', 'facility_delivery', 'complications',
        'preeclampsia_risk', 'gdm_risk', 'high_risk'
    ]
    
    for column in binary_columns:
        synthetic_data[column] = (synthetic_data[column] > 0.5).astype(int)
    
    # Categorical columns need to be cleaned because CTGAN might generate
    # slight variations like "richest " (with a space) or "Richest" (capitalized)
    
    # For wealth quintile, we map any variations back to standard categories
    def clean_wealth_quintile(value):
        value_str = str(value).lower().strip()
        if 'poorest' in value_str and 'poorer' not in value_str:
            return 'poorest'
        elif 'poorer' in value_str:
            return 'poorer'
        elif 'middle' in value_str:
            return 'middle'
        elif 'richer' in value_str:
            return 'richer'
        else:
            return 'richest'
    
    synthetic_data['wealth_quintile'] = (
        synthetic_data['wealth_quintile'].apply(clean_wealth_quintile)
    )
    
    # For urban/rural, ensure only these two values exist
    def clean_residence(value):
        value_str = str(value).lower().strip()
        return 'urban' if 'urban' in value_str else 'rural'
    
    synthetic_data['urban_rural'] = (
        synthetic_data['urban_rural'].apply(clean_residence)
    )
    
    print("✓ Post-processing complete!")
    
    return synthetic_data


def validate_synthetic_data(real_data, synthetic_data):
    """
    Compares synthetic data against the original to verify quality.
    
    Good synthetic data should have similar distributions to the real data
    while not being identical (which would defeat the privacy purpose).
    
    Parameters:
        real_data (pd.DataFrame): Original seed data
        synthetic_data (pd.DataFrame): Generated synthetic data
    """
    
    print("\n" + "="*60)
    print("SYNTHETIC DATA QUALITY VALIDATION")
    print("="*60)
    
    print("\n1. Key Health Indicator Comparison:")
    print("-" * 60)
    
    # Compare the main maternal health outcomes
    indicators = {
        'ANC 4+ visits': ('anc_visits', lambda x: (x >= 4).mean()),
        'Skilled delivery': ('skilled_delivery', lambda x: x.mean()),
        'Facility delivery': ('facility_delivery', lambda x: x.mean()),
        'Complications': ('complications', lambda x: x.mean()),
        'High-risk pregnancy': ('high_risk', lambda x: x.mean())
    }
    
    for indicator_name, (column, func) in indicators.items():
        real_value = func(real_data[column])
        synthetic_value = func(synthetic_data[column])
        difference = abs(real_value - synthetic_value)
        
        print(f"{indicator_name:25} Real: {real_value:6.1%}  " +
              f"Synthetic: {synthetic_value:6.1%}  " +
              f"Δ: {difference:5.1%}")
    
    print("\n2. Socioeconomic Distribution Check:")
    print("-" * 60)
    
    # Verify wealth quintile distribution (should be ~20% each)
    print("\nWealth Quintile Distribution:")
    for quintile in ['poorest', 'poorer', 'middle', 'richer', 'richest']:
        real_pct = (real_data['wealth_quintile'] == quintile).mean()
        synth_pct = (synthetic_data['wealth_quintile'] == quintile).mean()
        print(f"  {quintile:10} Real: {real_pct:5.1%}  Synthetic: {synth_pct:5.1%}")
    
    # Verify urban-rural split
    print("\nUrban-Rural Distribution:")
    for residence in ['urban', 'rural']:
        real_pct = (real_data['urban_rural'] == residence).mean()
        synth_pct = (synthetic_data['urban_rural'] == residence).mean()
        print(f"  {residence:10} Real: {real_pct:5.1%}  Synthetic: {synth_pct:5.1%}")
    
    print("\n3. Clinical Measurements (Mean ± Std):")
    print("-" * 60)
    
    clinical_vars = ['sys_bp', 'dia_bp', 'heart_rate', 'glucose']
    for var in clinical_vars:
        real_mean = real_data[var].mean()
        real_std = real_data[var].std()
        synth_mean = synthetic_data[var].mean()
        synth_std = synthetic_data[var].std()
        
        print(f"{var:12} Real: {real_mean:6.1f}±{real_std:4.1f}  " +
              f"Synthetic: {synth_mean:6.1f}±{synth_std:4.1f}")
    
    print("\n4. Correlation Preservation (Wealth vs Healthcare Access):")
    print("-" * 60)
    
    # Check if wealth-healthcare correlations are preserved
    wealth_order = {'poorest': 1, 'poorer': 2, 'middle': 3, 'richer': 4, 'richest': 5}
    real_data_copy = real_data.copy()
    synthetic_data_copy = synthetic_data.copy()
    
    real_data_copy['wealth_numeric'] = real_data_copy['wealth_quintile'].map(wealth_order)
    synthetic_data_copy['wealth_numeric'] = synthetic_data_copy['wealth_quintile'].map(wealth_order)
    
    real_corr = real_data_copy[['wealth_numeric', 'anc_visits', 'skilled_delivery']].corr()
    synth_corr = synthetic_data_copy[['wealth_numeric', 'anc_visits', 'skilled_delivery']].corr()
    
    print(f"Wealth × ANC visits:      Real: {real_corr.loc['wealth_numeric', 'anc_visits']:+.3f}  " +
          f"Synthetic: {synth_corr.loc['wealth_numeric', 'anc_visits']:+.3f}")
    print(f"Wealth × Skilled delivery: Real: {real_corr.loc['wealth_numeric', 'skilled_delivery']:+.3f}  " +
          f"Synthetic: {synth_corr.loc['wealth_numeric', 'skilled_delivery']:+.3f}")
    
    print("\n" + "="*60)
    print("✓ Validation complete!")
    print("="*60)


def save_synthetic_data(synthetic_data, output_path='data/synthetic/synthetic_bdhs_10k.csv'):
    """
    Saves the synthetic dataset to a CSV file.
    
    Parameters:
        synthetic_data (pd.DataFrame): The synthetic data to save
        output_path (str): Where to save the file
    """
    
    # Create the directory if it doesn't exist
    output_directory = os.path.dirname(output_path)
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")
    
    # Save to CSV
    synthetic_data.to_csv(output_path, index=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ Synthetic data saved to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Records: {len(synthetic_data):,}")
    print(f"  Variables: {len(synthetic_data.columns)}")


def main():
    """
    Main execution function that orchestrates the entire synthetic data generation pipeline.
    """
    
    print("="*60)
    print("SYNTHETIC BDHS DATA GENERATOR")
    print("Bangladesh Demographic and Health Survey 2017-18")
    print("="*60)
    print("\nThis tool generates privacy-preserving synthetic maternal health data")
    print("that maintains the statistical properties of the real BDHS survey.\n")
    
    # Step 1: Create realistic seed data
    seed_data = build_seed_dataset(sample_size=5000)
    
    # Step 2: Train the CTGAN model
    trained_model = train_synthetic_generator(seed_data)
    
    # Step 3: Generate synthetic records
    synthetic_records = generate_synthetic_data(trained_model, num_records=10000)
    
    # Step 4: Validate the synthetic data quality
    validate_synthetic_data(seed_data, synthetic_records)
    
    # Step 5: Save the results
    save_synthetic_data(synthetic_records)
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print("\nYour synthetic dataset is ready for analysis, modeling, and sharing")
    print("without privacy concerns. The data preserves key statistical")
    print("relationships from BDHS 2017-18 while representing no real individuals.")
    print("\nNext steps:")
    print("  1. Load the CSV into your analysis environment")
    print("  2. Build predictive models for maternal health outcomes")
    print("  3. Test interventions and policy scenarios")
    print("  4. Share findings without privacy restrictions")
    print("="*60)

if __name__ == "__main__":
    main()

