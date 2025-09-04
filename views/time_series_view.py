import matplotlib.pyplot as plt
import seaborn as sns

def plot_time_series(x, y, filename='', title='', dir_path='outputs/'):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label=f'{title}', color='tab:blue')
    plt.title(f'{title}')
    plt.xlabel('Data')
    plt.ylabel(f'{title}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    
    output_file = dir_path + 'tm_' + filename + '.png'
    plt.savefig(output_file)
    print(f"Salvando em: {output_file}")
    plt.close()

def plot_model_performance(X_pred, y_pred, X_true, y_true, filename='', title='', namespace='',
                            mape='', r2='', dir_path='outputs/'):
    plt.figure(figsize=(12, 6))
    plt.plot(X_true, y_true, label=f'{title} Real', color='tab:blue')
    plt.plot(X_pred, y_pred, label=f'{title} Previsto', color='tab:orange')
    plt.title(f'{title} - {namespace}')
    plt.xlabel('Data')
    plt.ylabel(f'{title}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower right')
    ax = plt.gca()
    ax.text(0.98, 0.98, f'MAPE: {mape}\nR2: {r2}',
            transform=ax.transAxes,
            fontsize=12,
            va='top', ha='right',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    
    output_file = f"{dir_path}p_{filename}_{namespace}.png"
    plt.savefig(output_file)
    print(f"Salvando em: {output_file}")
    plt.close()

def plot_model_evolution(X_pred, y_pred, X_true, y_true, filename='', title='', namespace='',
                            mape='', r2='', split_idx=None, dir_path='outputs/'):
    plt.figure(figsize=(12, 6))
    plt.plot(X_true, y_true, label=f'{title} Real', color='tab:blue')
    plt.plot(X_pred, y_pred, label=f'{title} Previsto', color='tab:orange')
    plt.title(f'{title} - {namespace}')
    plt.xlabel('Data')
    plt.ylabel(f'{title}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower right')

    if split_idx is not None:
        split_time = X_true.iloc[split_idx]
        plt.axvline(x=split_time, color='red', linestyle='--', linewidth=2, label='Início do teste')
        plt.legend(loc='lower right')

    ax = plt.gca()
    ax.text(0.01, 0.98, f'MAPE: {mape}\nR2: {r2}',
            transform=ax.transAxes,
            fontsize=12,
            va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    
    output_file = f"{dir_path}e_{filename}_{namespace}.png"
    plt.savefig(output_file)
    # plt.show()
    print(f"Salvando em: {output_file}")
    plt.close()

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