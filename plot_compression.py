import pandas as pd
import matplotlib.pyplot as plt

def generate_plot(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    # Filter numeric rows
    df_quant = df[df['slots'] != 'unquant'].copy()
    
    # Clean data
    df_quant = df_quant.dropna(subset=['compression_ratio', 'logit_round_decimals'])
    df_quant['slots'] = df_quant['slots'].astype(int)
    df_quant['logit_round_decimals'] = df_quant['logit_round_decimals'].astype(int)
    df_quant['compression_ratio'] = df_quant['compression_ratio'].astype(float)
    
    # 1. Only pick logit decimals = [10, 5, 1, 0]
    df_quant = df_quant[df_quant['logit_round_decimals'].isin([10, 5, 1, 0])]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 3]})
    fig.subplots_adjust(hspace=0.1)
    
    # Plot each slot configuration on both axes
    slots_sorted = sorted(df_quant['slots'].unique())
    for slot in slots_sorted:
        subset = df_quant[df_quant['slots'] == slot].groupby('logit_round_decimals')['compression_ratio'].mean().reset_index()
        subset = subset.sort_values('logit_round_decimals')
        ax1.plot(subset['logit_round_decimals'], subset['compression_ratio'], marker='o', label=f"{slot}")
        ax2.plot(subset['logit_round_decimals'], subset['compression_ratio'], marker='o', label=f"{slot}")
        
    # Baseline from unquantized
    unquant = df[df['slots'] == 'unquant']
    if not unquant.empty:
        baseline = unquant['compression_ratio'].dropna().astype(float).mean()
        ax1.axhline(y=baseline, color='red', linestyle='--', label="Unquantized")
        ax2.axhline(y=baseline, color='red', linestyle='--', label="Unquantized")
        
    # 2. Add gzip baseline
    gzip_baseline = 393 * 8 / 638
    ax1.axhline(y=gzip_baseline, color='green', linestyle=':', label="gzip")
    ax2.axhline(y=gzip_baseline, color='green', linestyle=':', label="gzip")
    
    # Zoom-in / limit the view to different portions of the data
    ax1.set_ylim(4.5, 5.2)  # outliers only
    ax2.set_ylim(0.2, 1.0)  # most of the data
    
    # Hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    
    # Add diagonal lines to indicate the break
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    ax2.set_xlabel('Logit Rounding Decimals')
    
    # Center y-label across the 2 subplots
    fig.text(0.04, 0.5, 'Compression Ratio', va='center', rotation='vertical')
    
    ax1.set_title('Compression Ratio vs. Quantisation Levels')
    
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Legend on ax2
    ax2.legend(title="Number of slots", loc='best')
    
    ax2.set_xticks([0, 1, 5, 10])
    plt.tight_layout()
    # Adjust layout to make room for common ylabel
    fig.subplots_adjust(left=0.1) 
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

generate_plot(
    '/homes/rm2092/LLM_Text_Compression_IIB_Project/results/roundtrip/combined_crossdev_hf_none_results.csv',
    '/homes/rm2092/LLM_Text_Compression_IIB_Project/results/roundtrip/compression_ratio_plot.png'
)
