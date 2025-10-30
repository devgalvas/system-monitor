import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib.patches import Patch

def atualizar_medias_csv(df):
    df_filtrado = df[~df["namespace"].isin(["ALL"])]

    medias_globais = df_filtrado.drop(columns=["namespace", "query"]).mean()

    linha_all_global = {"namespace": "ALL", "query": "ALL"}
    linha_all_global.update(medias_globais.to_dict())

    linhas_all_queries = []
    for query, grupo in df_filtrado.groupby("query"):
        medias_query = grupo.drop(columns=["namespace", "query"]).mean()
        linha = {"namespace": "ALL", "query": query}
        linha.update(medias_query.to_dict())
        linhas_all_queries.append(linha)

    linhas_media_namespace = []
    for ns, grupo in df_filtrado.groupby("namespace"):
        medias_ns = grupo.drop(columns=["namespace", "query"]).mean()
        linha = {"namespace": ns, "query": "ALL"}
        linha.update(medias_ns.to_dict())
        linhas_media_namespace.append(linha)

    df_sem_all = df[df["namespace"] != "ALL"]

    df_atualizado = pd.concat(
        [df_sem_all, pd.DataFrame(linhas_all_queries + [linha_all_global] + linhas_media_namespace)],
        ignore_index=True
    )

    return df_atualizado

def resultados_por_namespace(paths, query):
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df = df[~df["namespace"].isin(["smartoss-application-prod", "ALL", "openshift-storage"])]
        df = df[df["query"] == query]
        df = df.groupby("namespace", as_index=False).mean(numeric_only=True)
        dfs.append(df)

    namespaces = sorted(set(dfs[0]['namespace']).union(dfs[1]['namespace']))
    width = 0.35
    x = np.arange(len(namespaces))

    mape1 = [dfs[0].set_index('namespace').reindex(namespaces)['MAPE']]
    mape2 = [dfs[1].set_index('namespace').reindex(namespaces)['MAPE']]

    mape1 = np.array(mape1).flatten()
    mape2 = np.array(mape2).flatten()

    plt.figure(figsize=(12, 6))
    plt.xticks(x, namespaces, rotation=30, ha='right')
    plt.yticks(range(0,100,10))

    super_colors = ["tab:orange" if m2 > m1 else "lightgray" for m1, m2 in zip(mape1, mape2)]
    simple_colors = ["tab:blue" if m2 > m1 else "silver" for m1, m2 in zip(mape1, mape2)]

    for label, m1, m2 in zip(plt.gca().get_xticklabels(), mape1, mape2):
        if m2 > m1:
            label.set_alpha(1.0)   
        else:
            label.set_alpha(0.2) 


    simple_bar = plt.bar(x - width/2, mape1, width, label='Simple NN Model', color=simple_colors)
    super_bar = plt.bar(x + width/2, mape2, width, label='Super NN Model', color=super_colors)
    plt.bar_label(simple_bar, label_type='edge', fmt='%.2f', fontsize=7)
    plt.bar_label(super_bar, label_type='edge', fmt='%.2f', fontsize=7)

    legend_elements = [
        Patch(facecolor="tab:blue", label="Simple NN Model"),
        Patch(facecolor="tab:orange", label="Super NN Model")
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig('outputs/neural_network/resultados_por_namespace.png')

    

