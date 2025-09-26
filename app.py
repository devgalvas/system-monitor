import pandas as pd

caminho_csv = "outputs/neural_network/super_nn_results.csv"

# 1. Ler CSV
df = pd.read_csv(caminho_csv)

# 2. Filtrar (remover smartoss-application-prod e ALL)
df_filtrado = df[~df["namespace"].isin(["smartoss-application-prod", "ALL"])]

# 3. Recalcular médias globais (sem ALL e smartoss)
medias_globais = df_filtrado.drop(columns=["namespace", "query"]).mean()

linha_all_global = {"namespace": "ALL", "query": "ALL"}
linha_all_global.update(medias_globais.to_dict())

# 4. Médias por query
linhas_all_queries = []
for query, grupo in df_filtrado.groupby("query"):
    medias_query = grupo.drop(columns=["namespace", "query"]).mean()
    linha = {"namespace": "ALL", "query": query}
    linha.update(medias_query.to_dict())
    linhas_all_queries.append(linha)

# 5. Médias por namespace (sobre todas as queries)
linhas_media_namespace = []
for ns, grupo in df_filtrado.groupby("namespace"):
    medias_ns = grupo.drop(columns=["namespace", "query"]).mean()
    linha = {"namespace": ns, "query": "ALL"}
    linha.update(medias_ns.to_dict())
    linhas_media_namespace.append(linha)

# 6. Remover linhas antigas ALL e adicionar as novas
df_sem_all = df[df["namespace"] != "ALL"]

df_atualizado = pd.concat(
    [df_sem_all, pd.DataFrame(linhas_all_queries + [linha_all_global] + linhas_media_namespace)],
    ignore_index=True
)

# 7. Salvar no mesmo CSV
print(df_atualizado.to_string())
df_atualizado.to_csv(caminho_csv, index=False)

print("Linhas de média (ALL) atualizadas com sucesso!")
