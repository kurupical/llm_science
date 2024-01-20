import torch
from torch import nn
import math
import torch.nn.functional as F
import dataclasses
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import tqdm
import wandb
import os
import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime as dt
import transformers
import logging
from sentence_transformers import SentenceTransformer
import json

@dataclasses.dataclass  
class RetrievalConfig:
    
    experiment_name: str
    
    # arcface parameters
    s: float = 16
    m: float = 0.25
    out_features: int = 2200
    
    # model parameters
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    fc_after_pooling: bool = False
    batch_size: int = 32
    iters_to_accumulate: int = 1
    max_length: int = 512
    epochs: int = 30
    lr_bert: float = 2e-5
    lr_fc: float = 3e-4
    
    freeze_embeddings: bool = True
    freeze_layers: int = 4
    reinitialize_layers: int = 0
    
    # other
    dataset_path: str = "data/retrieval/20230903025102.csv"
    debug: bool = True    

def get_logger(
    output_dir: str,
):
    """
    logger を作成する. formatter は "%Y-%m-%d %H:%M:%S" で作成する.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # formatter
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    
    # handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    handler = logging.FileHandler(f"{output_dir}/log.txt", "w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
    


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s, m, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class RetrievalNet(nn.Module):
    def __init__(self, cfg: RetrievalConfig, logger: logging.Logger):
        super(RetrievalNet, self).__init__()
        self.model = AutoModel.from_pretrained(cfg.model_name, cache_dir="./cache")
        self.cfg = cfg
        self.logger = logger
        self.fc = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.arc_margin_product = ArcMarginProduct(
            in_features=self.model.config.hidden_size,
            out_features=self.cfg.out_features, 
            s=self.cfg.s,
            m=self.cfg.m
        )
        self._model_preprocess()

    def _model_preprocess(self):
        self.logger.info("freeze embeddings")
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

        self.logger.info(f"freeze {self.cfg.freeze_layers} layers")
        for layer in self.model.encoder.layer[:self.cfg.freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        
    def forward(
        self, 
        text: torch.Tensor, 
        attention_mask: torch.Tensor,
        label: torch.Tensor=None
    ):
        feature = self.model(text, attention_mask)[0]  # (batch_size, max_length, hidden_size)
        
        feature = feature.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(1) # (batch_size, hidden_size)
        if self.cfg.fc_after_pooling:
            feature = self.fc(feature)
        if label is None:
            return feature
        x = self.arc_margin_product(feature, label)
        return x, feature
    

class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: RetrievalConfig,
    ):
        self.texts = df["text"].values
        if "wiki_id" in df.columns:
            self.wiki_ids = df["wiki_id"].values
            self.include_wiki_ids = True
        else:
            self.include_wiki_ids = False
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        out = self.tokenizer(
            str(self.texts[idx]),
            padding="max_length",
            max_length=self.cfg.max_length,
            truncation=True,
            return_tensors="pt",
        )
        text = out["input_ids"][0]
        attention_mask = out["attention_mask"][0]
        
        if self.include_wiki_ids:
            label = torch.tensor(self.wiki_ids[idx], dtype=torch.long)
            return text, attention_mask, label
        return text, attention_mask
    

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    fold: int,
    epoch: int,  
):
    """
    1 epoch の学習を行う
    """
    model.train()
    loss_meter = AverageMeter()
    
    # gradient accumulation を行う
    scaler = torch.cuda.amp.GradScaler()
    
    tk = tqdm.tqdm(loader, desc=f"train epoch {epoch}")    
    
    for i, batch in enumerate(tk):
        text = batch[0].to(device)
        attention_mask = batch[1].to(device)
        label = batch[2].to(device)

        with torch.cuda.amp.autocast():
            output, _ = model(text, attention_mask, label)
            loss = nn.CrossEntropyLoss()(output, label)
        
            loss_meter.update(loss.item(), text.size(0))
            loss = loss / model.cfg.iters_to_accumulate

        scaler.scale(loss).backward()
        
        if ((i + 1) % model.cfg.iters_to_accumulate == 0) or (i == (len(loader) - 1)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        tk.set_postfix(
            loss=loss_meter.avg,
            lr=optimizer.param_groups[0]["lr"],
        )
        wandb.log({f"train_loss_fold{fold}": loss_meter.avg})
    return loss_meter.avg
        

def valid_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    fold: int,
    epoch: int,
    name: str,
):
    """
    1 epoch の検証を行う
    """
    model.eval()
    loss_meter = AverageMeter()

    tk = tqdm.tqdm(loader, desc=f"{name} epoch {epoch}")    
    
    for i, batch in enumerate(tk):
        text = batch[0].to(device)
        attention_mask = batch[1].to(device)
        label = batch[2].to(device)

        with torch.no_grad():
            output, feature = model(text, attention_mask, label)
            loss = nn.CrossEntropyLoss()(output, label)
        
            loss_meter.update(loss.item(), text.size(0))
            loss = loss / model.cfg.iters_to_accumulate
        
        loss_meter.update(loss.item(), text.size(0))
    # wandb にログを送る
    wandb.log({f"{name}_loss_{fold}": loss_meter.avg})
        
    return loss_meter.avg


def main(config: RetrievalConfig):
    try:
        if not torch.cuda.is_available():
            raise ValueError("GPUが使えません")
        output_dir = f"output/text_retrieval/{os.path.basename(__file__)}/{dt.now().strftime('%Y%m%d%H%M%S')}_{config.experiment_name}"
        os.makedirs(output_dir, exist_ok=True)
        logger = get_logger(output_dir)
        df = pd.read_csv(config.dataset_path)
        
        if config.debug:
            df = df.iloc[:100]
        
        kf = KFold(n_splits=5)
        df["wiki_id"] = df["wiki_id"].astype("category").cat.codes.values # Label Encoding
        
        wandb.init(
            project="llm_science_retrieval",
            name=config.experiment_name,
            reinit=True,
        )
        # config を wandb に保存する
        wandb.config.update(config)
        
        # dataclass config を保存する
        with open(f"{output_dir}/config.json", "w") as f:
            json.dump(dataclasses.asdict(config), f, indent=4)
        device = "cuda"
        for fold, (train_index, valid_index) in enumerate(kf.split(df)):

            logger.info(f"fold: {fold}")
            model = RetrievalNet(cfg=config, logger=logger).to("cuda")
            
            df_train = df.iloc[train_index].reset_index(drop=True)
            df_valid = df.iloc[valid_index].reset_index(drop=True)
            # DataLoader の作成
            train_dataset = RetrievalDataset(df_train, config)
            valid_dataset = RetrievalDataset(df_valid, config)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            
            # optimizer: AdamW
            optimizer = torch.optim.AdamW([
                {"params": model.fc.parameters(), "lr": config.lr_fc},
                {"params": model.model.parameters(), "lr": config.lr_bert},
                {"params": model.arc_margin_product.parameters(), "lr": config.lr_fc},
            ])
            
            # scheduler: linear warmup (transformer.get_linear_schedule_with_warmup)
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=100,
                num_training_steps=len(train_loader) * config.epochs,
            )
            
            # training and evaluation
            best_loss = np.inf
            for epoch in range(config.epochs):
                logger.info(f"epoch: {epoch}")
                train_one_epoch(model, train_loader, optimizer, scheduler, device, fold, epoch)
                loss_val = valid_one_epoch(model, valid_loader, device, fold, epoch, name="valid")
                
                if best_loss > loss_val:
                    logger.info(f"best loss: {best_loss} -> {loss_val}")
                    best_loss = loss_val
                    
                    torch.save(model.state_dict(), f"{output_dir}/model_fold{fold}.pth")
                    wandb.run.summary["best_loss"] = loss_val
                else:
                    logger.info(f"{loss_val} is not better than {best_loss}(test).")

            break
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
        wandb.finish()


if __name__ == "__main__":
    

    # for fc_after_pooling in [False]:
    #     for s in [16, 32, 48]:
    #         for m in [0.25, 0.5, 0.75]:
    #             for freeze_layers in [0, 2, 4, 5]:
    #                 config = RetrievalConfig(
    #                     experiment_name=f"minilm_fc_after_pooling_{fc_after_pooling}_m_{m}_s_{s}_freeze_layers_{freeze_layers}",
    #                     debug=False, 
    #                     s=s,
    #                     m=m,
    #                     freeze_layers=freeze_layers,
    #                 )
    #                 main(config)

    for fc_after_pooling in [False]:
        for s in [8, 12, 16]:
            for m in [0.15, 0.2, 0.25]:
                for freeze_layers in [0]:
                    config = RetrievalConfig(
                        experiment_name=f"minilm_fc_after_pooling_{fc_after_pooling}_m_{m}_s_{s}_freeze_layers_{freeze_layers}",
                        debug=False, 
                        s=s,
                        m=m,
                        freeze_layers=freeze_layers,
                    )
                    main(config)

    for fc_after_pooling in [False, True]:
        for freeze_embeddings in [True, False]:
            config = RetrievalConfig(
                experiment_name=f"minilm_fc_after_pooling_{fc_after_pooling}_freeze_embeddings_{freeze_embeddings}",
                debug=False, 
                fc_after_pooling=fc_after_pooling,
                freeze_embeddings=freeze_embeddings,
            )
            main(config)

    for freeze_layers in [6]:
        config = RetrievalConfig(
            experiment_name=f"mpnet_freeze_layers_{freeze_layers}",
            debug=False, 
            model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
            freeze_layers=freeze_layers,
        )
        main(config)

    for freeze_layers in [0]:
        config = RetrievalConfig(
            experiment_name=f"deberta_small_freeze_layers_{freeze_layers}",
            debug=False, 
            model_name="microsoft/deberta-v3-small",
            freeze_layers=freeze_layers,
        )
        main(config)
        
    for freeze_layers in [6, 8, 10]:
        config = RetrievalConfig(
            experiment_name=f"deberta_base_{fc_after_pooling}_m_{m}_s_{s}_freeze_layers_{freeze_layers}",
            debug=False, 
            model_name="microsoft/deberta-v3-base",
            freeze_layers=freeze_layers,
        )
        main(config)