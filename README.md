# System-monitor

Pipeline para análise de dados de monitoramento do OpenShift usando modelos de Machine Learning.

## Sumário
- Visão geral
- Estrutura do projeto
- Instalação rápida
- Banco de dados (Postgres + opcional SSH)
- Ingestão de CSV
- DataLoader (resumo)
- Models, Controllers, Views, Outputs e Params
- Como executar controllers

---

## Visão geral
O projeto é modularizado para separar responsabilidades:
- models/ → lógica dos modelos (pré-processamento, treino, predição).
- controllers/ → orquestração do pipeline (treino, teste, visualização).
- views/ → geração e salvamento de gráficos/tabelas.
- outputs/ → arquivos gerados (.png, .csv, modelos salvos).
- params/ → parâmetros e pesos salvos dos modelos.
- scripts principais: `app.py`, `importCSV.py`.

## Estrutura (resumida)
```
importCSV.py
controllers/
  multi_nn_controller.py
  no_training_controller.py
  results_controller.py
  simple_nn_controller.py
  super_nn_controller.py
  overview_controller.py
  xgboost_controller.py
models/
  base_model.py
  dataloader.py
  multi_nn_model.py
  simple_nn_model.py
  super_nn_model.py
  nn_model.py
  xgboost_model.py
views/
  base_view.py
  overview.py
  time_series_view.py
  hyperparameters_view.py
  results_view.py
params/
  super_nn/
outputs/
  neural_network/
  xgboost/
  decomposition_*
  samples_daily_*
  Top50Namespaces*
requirements.txt
README.md
```

## Instalação rápida
1. Clonar:
```bash
git clone https://github.com/LuizF14/system-monitor.git
cd system-monitor
```
2. Criar e ativar ambiente virtual:
```bash
python -m venv .venv        # ou python -m venv venv
source .venv/bin/activate   # ou source venv/bin/activate
```
3. Instalar dependências:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Banco de dados (Postgres)
Recomenda-se usar Docker para o Postgres. Exemplo de serviço (docker-compose):
```yaml
services:
  db:
    image: postgres:15
    container_name: postgres-db
    restart: always
    environment:
      POSTGRES_DB: nome_da_db
      POSTGRES_USER: seu_usuario
      POSTGRES_PASSWORD: sua_senha
    ports:
      - "5432:5432"
    volumes:
      - ./pgdata:/var/lib/postgresql/data
```
Variáveis de conexão a configurar em `.env`:
```
DB_HOST=localhost
DB_NAME=meu_banco
DB_USER=usuario
DB_PASS=senha
DB_TABLE=ocnr_dados

# Opcional: túnel SSH
SSH_HOST=servidor.remoto.com
SSH_PORT=22
SSH_USER=ssh_user
SSH_PASS=ssh_senha
LOCAL_PORT=5433
```

## Ingestão de CSV
Utilize `importCSV.py` para inserir dados no banco. Verifique o mapeamento de colunas esperado antes de rodar.

## DataLoader (models/dataloader.py) — resumo
Classe responsável por:
- Abrir túnel SSH (opcional) com redirecionamento de porta.
- Conectar ao Postgres.
- Executar queries e retornar pandas.DataFrame.
- Criar materialized views para acelerar consultas:
  - `createOverviewView()` → cria `ocnr_overview` com agregações por namespace/query.
  - `createNamespaceView(namespace)` → materialized view filtrada por namespace.
- Métodos principais: `start_ssh_tunnel()`, `connect_to_db()`, `query_to_db()`, `close()`.

## Models
- Todos os modelos herdam de `BaseModel` (interface comum).
- Métodos obrigatórios em BaseModel:
  - `preprocess()`
  - `train()`
  - `predict()`
  - `load()` / `save()` (quando aplicável)
- Implementações: `simple_nn_model`, `multi_nn_model`, `super_nn_model`, `xgboost_model`, etc.

## Controllers
- Orquestram DataLoader → Models → Views.
- Cada controller implementa, no mínimo, `run()`.
- Exemplos de execução:
```bash
python -m controllers.simple_nn_controller
python -m controllers.xgboost_controller
```

## Views
- Plotagem com matplotlib / seaborn.
- Salvam gráficos em `outputs/` como `.png` e tabelas `.csv`.
- Organização de arquivos por subpastas e padrão de nomes:
  <prefixo>_<periodo>_<modelo>_<tarefa>_<namespace>.png
  - Prefixos: `tm` (timeseries), `p` (performance), `e` (evolution), `l` (lag search)

## Outputs
Diretórios e finalidade:
- `outputs/neural_network/` → resultados de redes neurais
- `outputs/xgboost/` → resultados do XGBoost
- Arquivos na raiz de `outputs/` → análises gerais, decomposições e estatísticas

## Params
- `params/` contém pesos e parâmetros salvos dos modelos.
- Métodos `.save()` e `.load()` dos modelos usam este diretório para persistência.

## Boas práticas
- Use materialized views para acelerar análises repetidas.
- Versione parâmetros importantes em `params/`.
- Salve gráficos e tabelas em `outputs/` com nomes padronizados.

## Como contribuir / rodar localmente
1. Configurar `.env` com credenciais do DB (ou usar Postgres local via Docker).
2. Inserir/validar dados com `importCSV.py`.
3. Rodar controllers conforme necessidade:
```bash
python -m controllers.overview_controller
python -m controllers.results_controller
```

---