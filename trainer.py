import os
import json
import torch
import torchvision
from tqdm import tqdm

from vit import ViT as Model

def load_config(config_path : str) -> dict :
    with open(config_path, 'rb') as fh:
        data = json.load(fh)
    return data

def load_training_artifact(config : dict ):
    model = Model(config)

    # toy data can fit in mem, but not real data - relies on data sampler to load data
    tfms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.ConvertImageDtype(torch.float32),
        torchvision.transforms.Resize((config['img_dim'][-2], config['img_dim'][-1]), antialias=True),
    ])
    data_train = torchvision.datasets.MNIST(root=config['data']+'/train', train=True, download=True, transform=tfms)
    data_val = torchvision.datasets.MNIST(root=config['data']+'/val', train=False, download=True, transform=tfms)

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss = torch.nn.CrossEntropyLoss()

    return model, data_train, data_val, opt, loss

def get_dataloader(dataset : torch.utils.data.Dataset , config : dict ) -> torch.utils.data.DataLoader :
    dl = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['batch_size'],
            pin_memory=True,
            shuffle=True,
        )
    return dl

class Trainer(object):
    def __init__(self, 
                 model : torch.nn.Module, 
                 dataloader_train : torch.utils.data.DataLoader,
                 dataloader_val : torch.utils.data.DataLoader,
                 optimizer : torch.optim.Optimizer,
                 loss : callable,
                 device_id : str ,
                 config : dict):
        self.device_id = device_id
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.config = config
        self.loss = loss
        self.optimizer = optimizer
        self.epochs_run = 0
        self.best_val_loss = float('inf')
        self.model = model.to(device_id)
        if os.path.exists(config['save_path']):
            print("Loading snapshot")
            self._load_snapshot(config['save_path'])

    def _step(self, source : torch.Tensor , targets: torch.Tensor , pbar = None ) -> float :
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch : int ):
        b_sz = len(next(iter(self.dataloader_train))[0])
        pbar = tqdm(self.dataloader_train)
        pbar.set_description(f"[Dev {self.device_id}] Epoch {epoch} | Steps: {len(self.dataloader_train)}")
        self.model.train()
        loss, beta = 0.0, 0.5
        for t, (source, targets) in enumerate(pbar):
            source = source.to(self.device_id)
            targets = targets.to(self.device_id)
            l = self._step(source, targets, pbar)
            loss =  (beta*loss + (1-beta)*l) / (1-beta**(t+1))
            pbar.set_description(f"[Dev {self.device_id}] Epoch {epoch} | Steps: {len(self.dataloader_train)} | Train Loss: {loss:.04f}")

    def _save_snapshot(self, epoch):
        val_loss = 0.0
        print('Running val')
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(self.dataloader_val)
            for source, targets in pbar:
                source = source.to(self.device_id)
                targets = targets.to(self.device_id)
                output = self.model(source)
                l = self.loss(output, targets).item()
                val_loss += l

            val_loss /= len(self.dataloader_val)
            print(f'Val loss: {val_loss:.04f}, Best before this: {self.best_val_loss:.04f}')

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            snapshot = {
                "MODEL_STATE": self.model.state_dict(),
                "OPTIMIZER_STATE": self.optimizer.state_dict(),
                "VAL_LOSS": self.best_val_loss,
                "EPOCHS_RUN": epoch,
            }
            torch.save(snapshot, self.config['save_path'])
            print(f"Epoch {epoch} | Training snapshot saved at {self.config['save_path']}")

    def _load_snapshot(self, snapshot_path : str ):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.best_val_loss = snapshot["BEST_VAL_LOSS"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}, Val loss: {self.best_val_loss:.04f}")

    def train(self, ):
        for epoch in range(self.epochs_run, self.config['num_epochs']):
            self._run_epoch(epoch)
            if epoch % self.config['save_every'] == 0:
                self._save_snapshot(epoch)

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config = load_config('./config.json')
    model, data_train, data_val, opt, loss = load_training_artifact(config)
    dataloader_train = get_dataloader(data_train, config)
    dataloader_val = get_dataloader(data_val, config)
    trainer = Trainer(model, dataloader_train, dataloader_val, opt, loss, device, config)
    trainer.train()

if __name__ == '__main__':
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    train()