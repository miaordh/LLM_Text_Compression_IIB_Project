import pandas as pd
import matplotlib.pyplot as plt

def plot_matched_characters():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # LEFT SUBPLOT: cross-device
    df_cd = pd.read_csv('/homes/rm2092/LLM_Text_Compression_IIB_Project/results/roundtrip/combined_crossdev_hf_none_results.csv')
    df_cd['matching_characters'] = pd.to_numeric(df_cd['matching_characters'], errors='coerce').fillna(0)
    
    df_cd_quant = df_cd[df_cd['slots'] != 'unquant'].copy()
    df_cd_quant['slots'] = df_cd_quant['slots'].astype(int)
    df_cd_quant['logit_round_decimals'] = df_cd_quant['logit_round_decimals'].astype(int)
    df_cd_quant = df_cd_quant[df_cd_quant['logit_round_decimals'].isin([10, 5, 1, 0])]
    
    slots_sorted = sorted(df_cd_quant['slots'].unique())
    for slot in slots_sorted:
        subset = df_cd_quant[df_cd_quant['slots'] == slot].groupby('logit_round_decimals')['matching_characters'].mean().reset_index()
        subset = subset.sort_values('logit_round_decimals')
        ax1.plot(subset['logit_round_decimals'], subset['matching_characters'], marker='o', label=f"{slot}")

    ax1.axhline(y=626, color='red', linestyle='--', label="All-match (626)")

    ax1.set_title("Cross-device (CPU to GPU)")
    ax1.set_xlabel('Logit Rounding Decimals')
    ax1.set_ylabel('Matched Characters')
    ax1.set_xticks([0, 1, 5, 10])
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(title="Number of slots")

    # RIGHT SUBPLOT: cross-tp
    df_tp = pd.read_csv('/homes/rm2092/LLM_Text_Compression_IIB_Project/results/cross_tp_quant/cross_tp_quant_results_cblgpu11_3677468_1778572785959619962_with_matching.csv')
    df_tp['matching_characters'] = pd.to_numeric(df_tp['matching_characters'], errors='coerce').fillna(0)
    
    # filter determinism_mode None (empty or NaN)
    df_tp = df_tp[df_tp['determinism_mode'].isna()]
    
    df_tp_quant = df_tp[df_tp['quant'] == True].copy()
    df_tp_quant['slots'] = df_tp_quant['slots'].astype(int)
    df_tp_quant['logit_round_decimals'] = df_tp_quant['logit_round_decimals'].astype(int)
    df_tp_quant = df_tp_quant[df_tp_quant['logit_round_decimals'].isin([10, 5, 1, 0])]
    
    slots_sorted_tp = sorted(df_tp_quant['slots'].unique())
    for slot in slots_sorted_tp:
        subset = df_tp_quant[df_tp_quant['slots'] == slot].groupby('logit_round_decimals')['matching_characters'].mean().reset_index()
        subset = subset.sort_values('logit_round_decimals')
        ax2.plot(subset['logit_round_decimals'], subset['matching_characters'], marker='o', label=f"{slot}")

    ax2.axhline(y=626, color='red', linestyle='--', label="All-match (626)")

    ax2.set_title("Cross-TP (TP=1 to TP=2, Kernel = None)")
    ax2.set_xlabel('Logit Rounding Decimals')
    ax2.set_xticks([0, 1, 5, 10])
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(title="Number of slots")

    plt.suptitle("Matched Characters vs. Quantisation Levels")
    plt.tight_layout()
    output_path = '/homes/rm2092/LLM_Text_Compression_IIB_Project/results/matched_characters_plot.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

plot_matched_characters()
