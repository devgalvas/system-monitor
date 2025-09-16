import matplotlib.pyplot as plt
import seaborn as sns

def plot_lag_search(r2, mape, lags, filename='', title='', namespace='', dir_path='outputs/'):
    plt.figure(figsize=(12, 6))
    plt.plot(lags, r2, label=f'R2', color='tab:green')
    plt.plot(lags, mape, label=f'MAPE', color='tab:blue')
    plt.title(f'{title} - {namespace}')
    plt.xlabel('Lags')
    plt.ylabel(f'Performance')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower right')

    output_file = f"{dir_path}l_{filename}_{namespace}.png"
    plt.savefig(output_file)
    print(f"Salvando em: {output_file}")
    plt.close()

def plot_history(history, metrics, filename='', title='', namespace='', dir_path='outputs/'):
    epochs = range(1, len(history.history['loss']) + 1)
    # plt.plot(epochs, history.history['val_r2_score'], label='R2 (Validation)')
    plt.plot(epochs, history.history['val_mean_absolute_percentage_error'], label='MAPE (Validation)')
    # plt.axhline(metrics['R2'], label='R2 (Test)', color='red', linestyle='--', linewidth=2,)
    plt.axhline(metrics['MAPE'] * 100, label='MAPE (Test)', color='green', linestyle='--', linewidth=2,)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)

    output_file = f"{dir_path}h_{filename}_{namespace}.png"
    plt.savefig(output_file)
    print(f"Salvando em: {output_file}")
    plt.close()