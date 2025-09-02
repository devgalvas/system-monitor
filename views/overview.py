from views.base_view import BaseView
import matplotlib.pyplot as plt
import seaborn as sns

class Overview(BaseView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_means(self, df, top_n=50, title=''):
        df_top = df.head(top_n)

        plt.figure(figsize=(12, 6))
        palette_original = sns.color_palette('Reds', n_colors=len(df_top))
        palette_invertida = palette_original[::-1]
        sns.barplot(data=df_top, x='ocnr_tx_namespace', y='avg_result', palette=palette_invertida, hue='ocnr_tx_namespace')

        plt.title(f'Top {top_n} Namespaces por Média do {title}')
        plt.xlabel('Namespace')
        plt.ylabel(f'{title} Médio')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_file = self.dir_path + 'Top50NamespacesMeans.png'
        plt.savefig(output_file)
        plt.close()

    def plot_stdev(self, df, top_n=50, title=''):
        df_top = df.head(top_n)

        plt.figure(figsize=(12, 6))
        palette_original = sns.color_palette("Greens", n_colors=len(df_top))
        palette_invertida = palette_original[::-1]
        sns.barplot(data=df_top, x='ocnr_tx_namespace', y='stddev_result', palette=palette_invertida, hue='ocnr_tx_namespace')

        plt.title(f'Top {top_n} Namespaces por Desvio Padrão do {title}')
        plt.xlabel('Namespace')
        plt.ylabel(f'Desvio Padrão do {title}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_file = self.dir_path + 'Top50NamespacesStdev.png'
        plt.savefig(output_file)
        plt.close()

    def plot_volume_data(self, df, top_n=50):
        df_top = df.head(top_n)

        plt.figure(figsize=(12, 6))
        palette_original = sns.color_palette("Blues", n_colors=len(df_top))
        palette_invertida = palette_original[::-1]
        sns.barplot(data=df_top, x='ocnr_tx_namespace', y='count', palette=palette_invertida, hue='ocnr_tx_namespace')

        plt.title(f'Top {top_n} Namespaces por Volume de Dados')
        plt.xlabel('Namespace')
        plt.ylabel('Total de Registros')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_file = self.dir_path + 'Top50NamespacesVolumeDados.png'
        plt.savefig(output_file)
        plt.close()

    def plot_decomposition(self, result, filename='', namespace=''):
        fig = result.plot()
        output_file = f'{self.dir_path}decomposition_{filename}_{namespace}.png'
        plt.savefig(output_file)
        plt.close()

    def plot_samples_daily(self, df, filename='', namespace='', mean=0):
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='dia', y='total', color='dodgerblue')

        plt.title(f'Número de Samples por Dia - {namespace}')
        plt.xlabel('Dias')
        plt.ylabel('Número de samples')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.axhline(y=mean, color='red', linestyle='--', linewidth=2, label=f'Média = {mean}')

        plt.legend()

        output_file = f'{self.dir_path}samples_daily_{filename}_{namespace}.png'
        plt.savefig(output_file)
        plt.close()