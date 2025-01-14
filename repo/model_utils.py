import random, os
import numpy as np
import torch
from tqdm import tqdm
import wandb
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import json

def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_checkpoint(model, path):
    checkpoints = {
        "model_dict": model.state_dict(),
    }
    torch.save(checkpoints, path)


def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))["model_dict"])



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=3, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score + self.delta > self.best_score:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        checkpoints = {"model_dict": model.state_dict()}
        torch.save(checkpoints, self.path)
        self.val_loss_min = val_loss
        
from torch.utils.data import DataLoader
#_use_scheduler_lr
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

def to_float(x, device):
    if isinstance(x,list):
        list_x = []
        for x_i in x:
            x_i = x_i.to(device).float()
            list_x.append(x_i)
        x = list_x
    else:
        x = x.to(device).float()
        
    return x
 

def train_func(model, train_dataset, valid_dataset, early_stopping, loss_func, optimizer, args, device):
    model.train()
    model.to(device)
    print("------Start training")

    list_train_loss = []
    list_valid_loss = []

    # Initialize the StepLR scheduler
    if args._use_scheduler_lr:
        if args.scheduler_type == "steplr":
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Adjust step_size and gamma as needed
        elif args.scheduler_type == 'reducelronplateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        else:
            raise ValueError("scheduler")
        
    for epoch in range(args.epochs):
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        epoch_loss = []
        if not early_stopping.early_stop:
            model.train()
            for data in tqdm(train_dataloader):
                optimizer.zero_grad()
                x_train, y_train = to_float(data['x'], device), to_float(data['y'],device)
                y_ = model(x_train)
                loss = loss_func(y_.squeeze(), y_train.squeeze())

                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
            train_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            list_train_loss.append(train_epoch_loss)

            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            model.eval()
            with torch.no_grad():
                valid_epoch_loss = []
                for data in valid_dataloader:
                    x_train, y_train = to_float(data['x'], device), to_float(data['y'],device)
                    y_ = model(x_train)
                    loss = loss_func(y_.squeeze(), y_train.squeeze())
                    valid_epoch_loss.append(loss.item())
                valid_epoch_loss = sum(valid_epoch_loss) / len(valid_epoch_loss)
                list_valid_loss.append(valid_epoch_loss)

            early_stopping(valid_epoch_loss, model)

            # Step the scheduler every epoch
            if args._use_scheduler_lr:
                if args.scheduler_type == "steplr":
                    scheduler.step()
                elif args.scheduler_type == "reducelronplateau":
                    scheduler.step(valid_epoch_loss)
                else:
                    pass

            print(f"Training epoch {epoch} Train loss: {train_epoch_loss} Valid loss: {valid_epoch_loss}")
            if args._use_wandb:
                wandb.log({"loss/train_loss": train_epoch_loss,
                           "loss/valid_loss": valid_epoch_loss})

    return list_train_loss, list_valid_loss



def train_multioutput_func(model, train_dataset, valid_dataset, early_stopping, loss_func, optimizer, args, device):
    model.train()
    model.to(device)
    print("------Start training")

    list_train_loss = []
    list_valid_loss = []

    # Initialize the StepLR scheduler
    if args._use_scheduler_lr:
        if args.scheduler_type == "steplr":
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Adjust step_size and gamma as needed
        elif args.scheduler_type == 'reducelronplateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        else:
            raise ValueError("scheduler")
        
    for epoch in range(args.epochs):
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        epoch_loss = []
        if not early_stopping.early_stop:
            model.train()
            for data in tqdm(train_dataloader):
                optimizer.zero_grad()
                x_train, y_train = to_float(data['x'], device), to_float(data['y'],device)
                y_, list_output = model.forward_train(x_train)
                
                
                
                loss = loss_func(y_.squeeze(), y_train.squeeze())
                list_loss = []
                for output_ in list_output:
                    list_loss.append(loss_func(output_.squeeze(), y_train.squeeze()))
                    
                list_loss.append(loss)
                
                stacked_tensors = torch.stack(list_loss)
                mean_loss = torch.mean(stacked_tensors, dim=0)
                
                mean_loss.backward()
                optimizer.step()

                epoch_loss.append(mean_loss.item())
            train_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            list_train_loss.append(train_epoch_loss)

            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            model.eval()
            with torch.no_grad():
                valid_epoch_loss = []
                for data in valid_dataloader:
                    x_train, y_train = to_float(data['x'], device), to_float(data['y'],device)
                    y_ = model(x_train)
                    loss = loss_func(y_.squeeze(), y_train.squeeze())
                    valid_epoch_loss.append(loss.item())
                valid_epoch_loss = sum(valid_epoch_loss) / len(valid_epoch_loss)
                list_valid_loss.append(valid_epoch_loss)

            early_stopping(valid_epoch_loss, model)

            # Step the scheduler every epoch
            if args._use_scheduler_lr:
                if args.scheduler_type == "steplr":
                    scheduler.step()
                elif args.scheduler_type == "reducelronplateau":
                    scheduler.step(valid_epoch_loss)
                else:
                    pass

            print(f"Training epoch {epoch} Train loss: {train_epoch_loss} Valid loss: {valid_epoch_loss}")
            if args._use_wandb:
                wandb.log({"loss/train_loss": train_epoch_loss,
                           "loss/valid_loss": valid_epoch_loss})

    return list_train_loss, list_valid_loss


def create_optimizer(model, args):
    # Collect all parameters from list_embeder and list_head
    embedder_params = list(model.list_embeder.parameters())
    head_params = list(model.list_head.parameters())

    # Combine all parameters into a single list
    all_params = embedder_params + head_params

    # Create an optimizer (e.g., Adam) with all parameters
    optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=args.l2_coef)
    
    return optimizer

def train_multioutput_2stage_func(model, train_dataset, valid_dataset, early_stopping, loss_func, optimizer, args, device):
    model.train()
    model.to(device)
    print("------Start training")

    if args._use_scheduler_lr:
        if args.scheduler_type == "steplr":
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Adjust step_size and gamma as needed
        elif args.scheduler_type == 'reducelronplateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        else:
            raise ValueError("scheduler")
    import copy 
    
    
    early_stopping2 = copy.deepcopy(early_stopping)
    optimizer2 = create_optimizer(model, args)
    
    
    
    #Stage 1:
    list_train_loss = []
    list_valid_loss = []
    for epoch in range(args.epochs):
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        epoch_loss = []
        if not early_stopping2.early_stop:
            model.train()
            for data in tqdm(train_dataloader):
                optimizer2.zero_grad()
                x_train, y_train = to_float(data['x'], device), to_float(data['y'],device)
                list_y = model.forward_stage1(x_train)
                
                list_loss = []
                for y_ in list_y:
                    loss = loss_func(y_.squeeze(), y_train.squeeze())
                    list_loss.append(loss)
                stacked_tensors = torch.stack(list_loss)
                mean_loss = torch.mean(stacked_tensors, dim=0)
                mean_loss.backward()
                optimizer2.step()

                epoch_loss.append(mean_loss.item())
            train_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            list_train_loss.append(train_epoch_loss)

            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            model.eval()
            with torch.no_grad():
                valid_epoch_loss = []
                for data in valid_dataloader:
                    x_train, y_train = to_float(data['x'], device), to_float(data['y'],device)
                    y_ = model(x_train)
                    loss = loss_func(y_.squeeze(), y_train.squeeze())
                    valid_epoch_loss.append(loss.item())
                valid_epoch_loss = sum(valid_epoch_loss) / len(valid_epoch_loss)
                list_valid_loss.append(valid_epoch_loss)

            early_stopping2(valid_epoch_loss, model)
            print(f"Stage1: Training epoch {epoch} Train loss: {train_epoch_loss} Valid loss: {valid_epoch_loss}")
            if args._use_wandb:
                wandb.log({"stage1_loss/train_loss": train_epoch_loss,
                           "stage1_loss/valid_loss": valid_epoch_loss})

    load_model(model, f"output/{args.group_name}/checkpoint/{args.name}.pt")


    #Stage 2:
    # Initialize the StepLR scheduler
    
    list_train_loss = []
    list_valid_loss = []
    for epoch in range(args.epochs):
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        epoch_loss = []
        if not early_stopping.early_stop:
            model.train()
            for data in tqdm(train_dataloader):
                optimizer.zero_grad()
                x_train, y_train = to_float(data['x'], device), to_float(data['y'],device)
                y_ = model(x_train)
                loss = loss_func(y_.squeeze(), y_train.squeeze())

                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
            train_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            list_train_loss.append(train_epoch_loss)

            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            model.eval()
            with torch.no_grad():
                valid_epoch_loss = []
                for data in valid_dataloader:
                    x_train, y_train = to_float(data['x'], device), to_float(data['y'],device)
                    y_ = model(x_train)
                    loss = loss_func(y_.squeeze(), y_train.squeeze())
                    valid_epoch_loss.append(loss.item())
                valid_epoch_loss = sum(valid_epoch_loss) / len(valid_epoch_loss)
                list_valid_loss.append(valid_epoch_loss)

            early_stopping(valid_epoch_loss, model)

            # Step the scheduler every epoch
            if args._use_scheduler_lr:
                if args.scheduler_type == "steplr":
                    scheduler.step()
                elif args.scheduler_type == "reducelronplateau":
                    scheduler.step(valid_epoch_loss)
                else:
                    pass

            print(f"Stage2: Training epoch {epoch} Train loss: {train_epoch_loss} Valid loss: {valid_epoch_loss}")
            if args._use_wandb:
                wandb.log({"loss/train_loss": train_epoch_loss,
                           "loss/valid_loss": valid_epoch_loss})

    return list_train_loss, list_valid_loss

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
)

# def mdape(y_true, y_pred):
# 	return np.median((np.abs(np.subtract(y_true, y_pred)/ y_true))) * 100

def cal_acc(y_prd, y_grt):
    mae = mean_absolute_error(y_grt, y_prd)
    mse = mean_squared_error(y_grt, y_prd, squared=True)
    mape = mean_absolute_percentage_error(y_grt, y_prd)
    rmse = mean_squared_error(y_grt, y_prd, squared=False)
    corr = np.corrcoef(np.reshape(y_grt, (-1)), np.reshape(y_prd, (-1)))[0][1]
    r2 = r2_score(y_grt, y_prd)
    # mdape_ = mdape(y_grt,y_prd)
    return mae, mse, mape, rmse, r2, corr


def test_func(model, test_dataloader,criterion , args, besttrack_scaler,device):
    model.eval() 
    list_prd = []
    list_grt = []
    epoch_loss = 0
    model.to(device)
    
    with torch.no_grad():
        for data in test_dataloader:
            x_train, y_grt = to_float(data['x'], device), to_float(data['y'],device)
            y_prd = model(x_train)
            batch_loss = criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
            y_prd = y_prd.cpu().detach().numpy()
            y_grt = y_grt.cpu().detach().numpy()
            if args.transform_groundtruth:
                y_prd = besttrack_scaler.inverse_transform(y_prd)
                y_grt = y_grt.reshape((y_grt.shape[0],1))
                y_grt = besttrack_scaler.inverse_transform(y_grt)
            
            y_prd = np.squeeze(y_prd).tolist()
            y_grt = np.squeeze(y_grt).tolist()
            list_prd += y_prd
            list_grt += y_grt
            epoch_loss += batch_loss.item()
    mae, mse, mape, rmse, r2, corr_ = cal_acc(list_prd, list_grt)
    
    return list_prd, list_grt, epoch_loss, mae, mse, mape, rmse, r2, corr_


def fit_scalers_in_batches(args, batch_size=100):
    """
    Args:
        args: Argument object containing the path to the data directory.
        batch_size: Number of samples per batch for incremental fitting.
    
    Returns:
        scaler: MinMaxScaler for input data.
        bt_scaler: MinMaxScaler for target data.
        n_fts: List of feature dimensions for input data.
    """

    scaler = MinMaxScaler()
    bt_scaler = MinMaxScaler()
    

    if os.path.exists(f"{args.data_dir}/scaler/scaler.pkl"):
        print("Load scaler")
        with open(f"{args.data_dir}/scaler/x_shape.json", "r") as f:
            x_shape = json.load(f)

        with open(f"{args.data_dir}/scaler/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        with open(f"{args.data_dir}/scaler/bt_scaler.pkl", "rb") as f:
            bt_scaler = pickle.load(f)
        
        if len(x_shape) == 4:  # 4D input: (batch, height, width, channels)
                n_fts = [x_shape[1]]
        elif len(x_shape) == 5:  # 5D input: (batch, time, height, width, channels)
            n_fts = [x_shape[2], x_shape[1]]
        else:
            raise ValueError(f"Unsupported input shape: {x_shape}")
    else:
        print("Cal scaler")
        data_dir_train = f"{args.data_dir}/train/data.npz"
        arr_train = np.load(data_dir_train)

        x_train, y_train = arr_train['x_arr'], arr_train['groundtruth']
        y_train *= 0.5

        x_shape = x_train.shape
        y_train_reshaped = y_train.reshape(-1, 1)
        bt_scaler.fit(y_train_reshaped)  # Fit the target scaler

        print("X shape", x_shape)
        if len(x_shape) == 4:
            # For 4D input: (batch, height, width, channels)
            x_train = x_train.transpose((0, 2, 3, 1))
            x_train_reshaped = x_train.reshape(-1, x_shape[1])
            n_fts = [x_train.shape[-1]]
        elif len(x_shape) == 5:
            # For 5D input: (batch, time, height, width, channels)
            x_train = x_train.transpose((0, 1, 3, 4, 2))
            x_train_reshaped = x_train.reshape(-1, x_shape[2])
            n_fts = [x_train.shape[-1], x_train.shape[1]]
        else:
            raise ValueError(f"Unsupported input shape: {x_shape}")

        # Reshape target data

        # Fit scalers
        print("X_train_reshaped", x_train_reshaped.shape)
        x_train_scaled = scaler.fit_transform(x_train_reshaped)
        os.makedirs(f"{args.data_dir}/scaler", exist_ok=True)
        with open(f"{args.data_dir}/scaler/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        
        with open(f"{args.data_dir}/scaler/bt_scaler.pkl", "wb") as f:
            pickle.dump(bt_scaler,f)
        
        with open(f"{args.data_dir}/scaler/x_shape.json", "w") as f:
            x_shape = json.dump(x_shape,f)
    print("Preprocess Done!!!")
    return scaler, bt_scaler, n_fts


def get_scaler():
    data_dir_train = "cutted_data/train/train_data.npz"
    arr_train = np.load(data_dir_train)
    x_train, y_train = arr_train['x_arr'], arr_train['groundtruth']
    x_train = x_train[:, :, 31, 31]
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train_reshaped)
    return scaler

from sklearn.preprocessing import MinMaxScaler

def get_scaler2(args):
    """
    Function to compute scalers and feature dimensions for training data.
    Args:
        args: Argument object containing the path to data directory.

    Returns:
        scaler: MinMaxScaler for input data.
        bt_scaler: MinMaxScaler for target data.
        n_fts: List of feature dimensions for input data.
    """
    # Initialize scalers
    scaler = MinMaxScaler()
    bt_scaler = MinMaxScaler()

    # Load training data
    data_dir_train = f"{args.data_dir}/train/data.npz"
    arr_train = np.load(data_dir_train)
    x_train, y_train = arr_train['x_arr'], arr_train['groundtruth']

    # Scale target data
    y_train *= 0.5

    # Reshape and prepare input data
    x_shape = x_train.shape
    if len(x_shape) == 4:
        # For 4D input: (batch, height, width, channels)
        x_train = x_train.transpose((0, 2, 3, 1))
        x_train_reshaped = x_train.reshape(-1, x_shape[1])
        n_fts = [x_train.shape[-1]]
    elif len(x_shape) == 5:
        # For 5D input: (batch, time, height, width, channels)
        x_train = x_train.transpose((0, 1, 3, 4, 2))
        x_train_reshaped = x_train.reshape(-1, x_shape[4])
        n_fts = [x_train.shape[-1], x_train.shape[1]]
    else:
        raise ValueError(f"Unsupported input shape: {x_shape}")

    # Reshape target data
    y_train_reshaped = y_train.reshape(-1, 1)

    # Fit scalers
    x_train_scaled = scaler.fit_transform(x_train_reshaped)
    y_train_scaled = bt_scaler.fit_transform(y_train_reshaped)

    return scaler, bt_scaler, n_fts

def train_func2(model, train_dataset, valid_dataset, early_stopping, loss_func, optimizer, args, device):
    model.train()
    model.to(device)
    print("------Start training")

    list_train_loss = []
    list_valid_loss = []

    # Initialize the StepLR scheduler
    if args._use_scheduler_lr:
        if args.scheduler_type == "steplr":
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Adjust step_size and gamma as needed
        elif args.scheduler_type == 'reducelronplateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        else:
            raise ValueError("scheduler")
        
    for epoch in range(args.epochs):
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        epoch_loss = []
        if not early_stopping.early_stop:
            model.train()
            for data in tqdm(train_dataloader):
                optimizer.zero_grad()
                x_train, y_train, leading_time = to_float(data['x'], device), to_float(data['y'],device), to_float(data['leading_time'], device)

                y_ = model((x_train,leading_time))
                loss = loss_func(y_.squeeze(), y_train.squeeze())

                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
            train_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            list_train_loss.append(train_epoch_loss)

            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            model.eval()
            with torch.no_grad():
                valid_epoch_loss = []
                for data in valid_dataloader:
                    x_train, y_train, leading_time = to_float(data['x'], device), to_float(data['y'],device), data['leading_time'].to(device)
                    y_ = model((x_train,leading_time))
                    loss = loss_func(y_.squeeze(), y_train.squeeze())
                    valid_epoch_loss.append(loss.item())
                valid_epoch_loss = sum(valid_epoch_loss) / len(valid_epoch_loss)
                list_valid_loss.append(valid_epoch_loss)

            early_stopping(valid_epoch_loss, model)

            # Step the scheduler every epoch
            if args._use_scheduler_lr:
                if args.scheduler_type == "steplr":
                    scheduler.step()
                elif args.scheduler_type == "reducelronplateau":
                    scheduler.step(valid_epoch_loss)
                else:
                    pass

            print(f"Training epoch {epoch} Train loss: {train_epoch_loss} Valid loss: {valid_epoch_loss}")
            if args._use_wandb:
                wandb.log({"loss/train_loss": train_epoch_loss,
                           "loss/valid_loss": valid_epoch_loss})

    return list_train_loss, list_valid_loss



def test_func2(model, test_dataloader,criterion , args, besttrack_scaler,device):
    model.eval() 
    list_prd = []
    list_grt = []
    epoch_loss = 0
    model.to(device)
    
    with torch.no_grad():
        for data in test_dataloader:
            x_train, y_grt, leading_time = to_float(data['x'], device), to_float(data['y'],device),to_float(data['leading_time'], device)
            y_prd = model((x_train, leading_time))

            batch_loss = criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
            y_prd = y_prd.cpu().detach().numpy()
            y_grt = y_grt.cpu().detach().numpy()
            if args.transform_groundtruth:
                y_prd = besttrack_scaler.inverse_transform(y_prd)
                y_grt = y_grt.reshape((y_grt.shape[0],1))
                y_grt = besttrack_scaler.inverse_transform(y_grt)
            
            y_prd = np.squeeze(y_prd).tolist()
            y_grt = np.squeeze(y_grt).tolist()
            list_prd += y_prd
            list_grt += y_grt
            epoch_loss += batch_loss.item()
    mae, mse, mape, rmse, r2, corr_ = cal_acc(list_prd, list_grt)
    
    return list_prd, list_grt, epoch_loss, mae, mse, mape, rmse, r2, corr_
