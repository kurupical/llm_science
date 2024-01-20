
import dataclasses
import torch
import pandas as pd
from sklearn.model_selection import KFold
from transformers import AutoModel, AutoTokenizer
import transformers
import wandb
from datetime import datetime as dt

import logging 

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
    
@dataclasses.dataclasses
class BertConfig:
    
    experiment_name: str
    lr_fc: float = 1e-4
    lr_bert: float = 1e-5
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 512
    batch_size: int = 16
    epochs: int = 20
    
    dataset_name: str    


class MultipleChoiceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: BertConfig,
    ):
        self.df = df
        self.cfg = cfg
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ret = {}
        for choice in ["A", "B", "C", "D", "E"]:
            text = row["prompt"] + " [SEP] " + row[choice]
            text = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.cfg.max_length,
                truncation=True,
                return_tensors="pt",
            )
            ret[choice] = text
        ret["label"] = row["answer"].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4})
        ret["id"] = row["id"]
        return ret
            

class BertModel(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
    ):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.fc = torch.nn.Linear(768, 1)
    
    def _forward_prompt(
        self,
        prompt: torch.Tensor,
    ):
        x = self.model(**prompt)  # (batch_size, max_length, 768)
        x = x.mean(dim=1)  # (batch_size, 768)
        x = self.fc(x)  # (batch_size, 1)
        return x
    
    def forward(
        self,
        prompt_a: torch.Tensor,
        prompt_b: torch.Tensor,
        prompt_c: torch.Tensor,
        prompt_d: torch.Tensor,
        prompt_e: torch.Tensor,
    ):
        a = self._forward_prompt(prompt_a)
        b = self._forward_prompt(prompt_b)
        c = self._forward_prompt(prompt_c)
        d = self._forward_prompt(prompt_d)
        e = self._forward_prompt(prompt_e)
        
        x = torch.cat([a, b, c, d, e], dim=1)  # (batch_size, 5)
        return x


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
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    fold: int,
    device: torch.device,
):
    """
    1 epoch の学習を行う
    """
    model.train()
    loss_meter = AverageMeter()
    
    for batch in loader:
        prompt_a = batch["A"]["input_ids"].to(device)
        prompt_b = batch["B"]["input_ids"].to(device)
        prompt_c = batch["C"]["input_ids"].to(device)
        prompt_d = batch["D"]["input_ids"].to(device)
        prompt_e = batch["E"]["input_ids"].to(device)
        label = batch["label"].to(device)
        
        optimizer.zero_grad()
        output = model(prompt_a, prompt_b, prompt_c, prompt_d, prompt_e)
        loss = torch.nn.BCEWithLogitsLoss()(output, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_meter.update(loss.item(), prompt_a.size(0))
        # wandb にログを送る
        wandb.log({f"train_loss_fold{fold}": loss_meter.avg})
    
def mapk(y_true, y_pred, k=3):
    """
    y_true: (n_samples, n_classes)
    y_pred: (n_samples, n_classes)
    """
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) == 2
    
    n_samples, n_classes = y_true.shape
    map_score = 0
    for i in range(n_samples):
        y_true_ = y_true[i]
        y_pred_ = y_pred[i]
        idx = np.argsort(y_pred_)[::-1]
        y_true_ = y_true_[idx]
        score = 0
        for j in range(k):
            score += y_true_[j] / (j + 1)
        score /= np.min([n_classes, k])
        map3 += score
    map_score /= n_samples
    return map_score


def valid_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    fold: int,
    name: str,
):
    """
    1 epoch の検証を行う
    """
    model.eval()
    loss_meter = AverageMeter()
    
    outputs = []
    
    for batch in loader:
        prompt_a = batch["A"]["input_ids"].to(device)
        prompt_b = batch["B"]["input_ids"].to(device)
        prompt_c = batch["C"]["input_ids"].to(device)
        prompt_d = batch["D"]["input_ids"].to(device)
        prompt_e = batch["E"]["input_ids"].to(device)
        label = batch["label"].to(device)
        
        with torch.no_grad():
            output = model(prompt_a, prompt_b, prompt_c, prompt_d, prompt_e)
            loss = torch.nn.BCEWithLogitsLoss()(output, label)
        
        outputs.append(torch.sigmoid(output).cpu().numpy())
        loss_meter.update(loss.item(), prompt_a.size(0))
        # wandb にログを送る
        wandb.log({f"{name}_loss_{fold}": loss_meter.avg})
    
    # map@3 を計算する
    outputs = torch.cat(outputs, dim=0)
    outputs = outputs.cpu().numpy()
    labels = df_valid["answer"].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}).values
    map3 = mapk([labels], outputs, k=3)
    wandb.log({f"{name}_map3_{fold}": map3})
    
    df_valid = pd.DataFrame({
        "pred_A": outputs[:, 0],
        "pred_B": outputs[:, 1],
        "pred_C": outputs[:, 2],
        "pred_D": outputs[:, 3],
        "pred_E": outputs[:, 4],
    })
    
    return df_valid, map3
    
    
def main(config: Config):
    model = BertModel(config.model_name)
    
    output_dir = f"output/exp001/{dt.now().strftime('%Y%m%d%H%M%S')}"
    df = pd.read_csv("")
    df_test = pd.read_csv("")

    kfold = KFold(5, random_state=42, shuffle=True)
    
    wandb.init(
        project="llm_science",
        name=config.experiment_name,
        reinit=True
    )

    test_dataset = MultipleChoiceDataset(df_test, config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    logger = get_logger(output_dir)
    for fold, (train_index, valid_index) in enumerate(kfold.split(df)):

        logger.info(f"fold: {fold}")
        df_train = df.iloc[train_index].reset_index(drop=True)
        df_valid = df.iloc[valid_index].reset_index(drop=True)
        # DataLoader の作成
        train_dataset = MultipleChoiceDataset(df_train, config)
        valid_dataset = MultipleChoiceDataset(df_valid, config)
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
        ])
        
        # scheduler: linear warmup (transformer.get_linear_schedule_with_warmup)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=len(train_loader) * config.epochs,
        )
        
        # training and evaluation
        best_score = 0
        for epoch in range(config.epochs):
            logger.info(f"epoch: {epoch}")
            train_one_epoch(model, train_loader, optimizer, scheduler, device, fold)
            df_valid_pred, _ = valid_one_epoch(model, valid_loader, device, fold, name="valid")
            df_test_pred, score_test = valid_one_epoch(model, test_loader, device, fold, name="test")
            
            # best model を保存する
            if best_score < score_test:
                logger.info(f"best score: {best_score} -> {score_test}")
                best_score = score_test
                torch.save(model.state_dict(), f"{output_dir}/model_fold{fold}.pth")
                pd.concat([df_valid, df_valid_pred], axis=1).to_csv(f"{output_dir}/valid_fold{fold}.csv", index=False)
                pd.concat([df_test, df_test_pred], axis=1).to_csv(f"{output_dir}/test_fold{fold}.csv", index=False)
            else:
                logger.info(f"{score_test} is not better than {best_score}.")