import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def test_drift_simulation():
    
    print("Testing Drift Detection...")
    print("=" * 50)
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    reference_data = np.random.normal(0, 1, (n_samples, n_features))
    
    test1_data = np.random.normal(0, 1, (500, n_features))
    
    test2_data = np.random.normal(0.2, 1, (500, n_features))
    
    test3_data = np.random.normal(1.0, 2.0, (500, n_features))
    
    test4_data = np.random.normal(3.0, 3.0, (500, n_features))
    
    def kolmogorov_smirnov_test(ref, test):
        ks_stats = []
        for i in range(n_features):
            ks_stat, p_value = stats.ks_2samp(ref[:, i], test[:, i])
            ks_stats.append(ks_stat)
        return np.mean(ks_stats)
    
    def calculate_psi(ref, test, bins=20):
        psi_vals = []
        for i in range(n_features):
            ref_min, ref_max = np.min(ref[:, i]), np.max(ref[:, i])
            bin_edges = np.linspace(ref_min, ref_max, bins + 1)
            
            ref_counts, _ = np.histogram(ref[:, i], bins=bin_edges)
            test_counts, _ = np.histogram(test[:, i], bins=bin_edges)
            
            ref_proportions = (ref_counts + 0.001) / (len(ref) + bins * 0.001)
            test_proportions = (test_counts + 0.001) / (len(test) + bins * 0.001)
            
            psi = 0
            for j in range(len(ref_proportions)):
                if ref_proportions[j] > 0 and test_proportions[j] > 0:
                    psi += (test_proportions[j] - ref_proportions[j]) * np.log(test_proportions[j] / ref_proportions[j])
            psi_vals.append(psi)
        return np.mean(psi_vals)
    
    tests = [test1_data, test2_data, test3_data, test4_data]
    test_names = ["No Drift", "Minor Drift", "Major Drift", "Extreme Drift"]
    
    results = []
    for test_name, test_data in zip(test_names, tests):
        ks_score = kolmogorov_smirnov_test(reference_data, test_data)
        psi_score = calculate_psi(reference_data, test_data)
        results.append({
            'Test': test_name,
            'KS Statistic': ks_score,
            'PSI': psi_score,
            'KS Drift (thresh=0.15)': ks_score > 0.15,
            'PSI Drift (thresh=0.25)': psi_score > 0.25
        })
    
    results_df = pd.DataFrame(results)
    print("\nDrift Detection Test Results:")
    print(results_df.to_string(index=False))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, (test_name, test_data) in enumerate(zip(test_names, tests)):
        row = idx // 2
        col = idx % 2
        
        axes[row, col].hist(reference_data[:, 0], alpha=0.5, label='Reference', bins=30, density=True)
        axes[row, col].hist(test_data[:, 0], alpha=0.5, label=test_name, bins=30, density=True)
        axes[row, col].set_title(f'{test_name}\nKS: {results[idx]["KS Statistic"]:.3f}, PSI: {results[idx]["PSI"]:.3f}')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.suptitle('Drift Detection Simulation Test', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('drift_test_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("OK")
    
    return results_df

if __name__ == "__main__":
    test_drift_simulation()