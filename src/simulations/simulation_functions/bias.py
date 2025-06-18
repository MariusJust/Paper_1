import numpy as np
from .Simulate_data import simulate, Pivot

from models import MultivariateModelGlobal as Model    



def calculate_bias(predictions, specification, best_node, cfg):

    true_data= simulate(
        seed=0,
        n_countries=196,
        n_years=63,
        specification=specification,
        add_noise=False
    )
    
    train_data= simulate(
        seed=0,
        n_countries=196,
        n_years=63,
        specification=specification,
        add_noise=True
    )
    
    weights_lists = list(zip(*predictions))

            # Now average across the rep dimension for each layer
    avg_weights = [
                np.mean(np.stack(list, axis=0), axis=0)
                for list in weights_lists
            ]
    growth, precip, temp = Pivot(train_data)
    x_train = {0:temp, 1:precip}
        
    factory = Model(nodes=best_node, x_train=x_train, y_train=growth, dropout=cfg.dropout, formulation=cfg.formulation, penalty=cfg.penalty)
        
    ensemble_model = factory.get_model()
    ensemble_model.model.set_weights(avg_weights)
    ensemble_model.fit(lr=cfg.lr, min_delta=cfg.min_delta, patience=cfg.patience, verbose=cfg.verbose)
   
    preds= np.squeeze(np.reshape(ensemble_model.in_sample_predictions(), (1,1,-1)))
    bias=np.sqrt(np.nanmean(preds- true_data["delta_logGDP"]) ** 2)
    
    return bias