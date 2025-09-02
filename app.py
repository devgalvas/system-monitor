from controllers.nn_controller import NeuralNetworkController
from controllers.xgboost_controller import XGBoostController
from controllers.overview_controller import OverviewController
from models.dataloader import DataLoader

if __name__ == "__main__":
    dl = DataLoader()
    dl.connect_to_db()
    # OverviewController(dataloader=dl, namespace=['panda-nifi', 'panda-druid', 'telefonica-cem-mbb-prod']).run()
    NeuralNetworkController(dataloader=dl, namespace=['panda-nifi', 'panda-druid', 'telefonica-cem-mbb-prod']).run()
    # XGBoostController(dataloader=dl, 
    #                   namespace=['panda-nifi', 'panda-druid', 'telefonica-cem-mbb-prod']
    #                   ).run()

    dl.close()
