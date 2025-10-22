import views.results_view as rs

import pandas as pd

if __name__ == "__main__":
    rs.atualizar_medias_csv("outputs/neural_network/super_nn_results.csv")
    rs.resultados_por_namespace(["outputs/neural_network/simple_nn_results.csv", 
                                 "outputs/neural_network/super_nn_results.csv"],
                                 query="ALL")