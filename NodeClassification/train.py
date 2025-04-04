# I need to call this file from the main.py file to train the model and get the results
import os
import random
import time
import numpy as np
import torch
from tqdm import tqdm



def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def train_and_get_results(data, model, p, lr, seed, splits, weight_decay=5e-4):

    def train():
        model.train()
        optimizer.zero_grad()  
        out = model(data.x, data.edge_index,p)          
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()  
        optimizer.step()  
        pred = out.argmax(dim=1)  
        train_correct = pred[train_mask] == data.y[train_mask]  
        train_acc = int(train_correct.sum()) / int(train_mask.sum())  
        return loss


    def val():
        model.eval()
        out = model(data.x, data.edge_index,p)
        pred = out.argmax(dim=1)  # Use the class with highest probability. 
        val_correct = pred[val_mask] == data.y[val_mask]  # Check against ground-truth labels.
        val_acc = int(val_correct.sum()) / int(val_mask.sum())  # Derive ratio of correct predictions.
        return val_acc


    def test():
            model.eval()
            out= model(data.x, data.edge_index,p)
            pred = out.argmax(dim=1)  # Use the class with highest probability. 
            test_correct = pred[test_mask] == data.y[test_mask]  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / int(test_mask.sum())  # Derive ratio of correct predictions.
            return test_acc

    test_acc_allsplits = []
    val_acc_allsplits = []
    for split_idx in range(splits):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        
        print(f"Training for split = {split_idx + 1}")
        
        train_mask = data.train_mask[:, split_idx]
        test_mask = data.test_mask[:, split_idx]
        val_mask = data.val_mask[:, split_idx]
        
        set_seed(seed)
        print("Start training ....")
                
        for epoch in tqdm(range(1, 101)):
            loss = train()
        val_acc = val()            
        test_acc = test()
        final_test_acc = test_acc * 100
        final_val_acc = val_acc*100
        test_acc_allsplits.append(final_test_acc)
        val_acc_allsplits.append(final_val_acc)
        print()
        print(f"Split {split_idx + 1}: Test Accuracy: {final_test_acc:.2f}%, Validation Accuracy: {final_val_acc:.2f}%")
    
    print(len(test_acc_allsplits))
    print()
    print(f"Average Test Accuracy: {np.mean(test_acc_allsplits):.2f}% ± {2 * np.std(test_acc_allsplits) / np.sqrt(len(test_acc_allsplits)):.2f}%")
    print(f"Average Validation Accuracy: {np.mean(val_acc_allsplits):.2f}% ± {2 * np.std(val_acc_allsplits) / np.sqrt(len(val_acc_allsplits)):.2f}%")

    return np.mean(test_acc_allsplits), 2 * np.std(test_acc_allsplits) / np.sqrt(len(test_acc_allsplits)),np.mean(val_acc_allsplits),2 * np.std(val_acc_allsplits) / np.sqrt(len(val_acc_allsplits))



