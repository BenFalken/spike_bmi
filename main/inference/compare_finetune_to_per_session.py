import torch
import matplotlib.pyplot as plt
import scipy, os
import numpy as np

EXP = 'bmi'
subjects = ['indy']
strategies = ['full_cohort_finetuned_optimal', 'per_session'] #full_cohort_finetuned_medium_soft
losses = {strategy: {} for strategy in strategies}

for sub in subjects:
    for strategy in strategies:
        finetune_path = f"/users/bfalkenb/scratch/bfalkenb/data/snn_checkpoints/{EXP}/{sub}/{strategy}"
        sessions = os.listdir(finetune_path)
        strat_losses = []
        for session in sessions:
            checkpoint = torch.load(f"{finetune_path}/{session}/best_model_weights.pth")
            loss = checkpoint['best_loss']
            strat_losses.append(loss)
        
        losses[strategy][sub] = strat_losses

        plt.hist(strat_losses, alpha=0.5, label=strategy)
    plt.title(f'{sub}')
    plt.savefig(f'compare_finetune_to_per_session_{sub}.png', dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show() 
    plt.close()


for sub in subjects:
    print(f'Subject {sub}')
    for strategy in strategies:
        print(f'Mean RMSE ({strategy}): {np.mean(losses[strategy][sub])}')
    t_test = scipy.stats.ttest_ind(losses[strategies[0]][sub], losses[strategies[1]][sub])
    print(f'T-test: {t_test}')
