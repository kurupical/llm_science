
import dataclasses
import torch
import pandas as pd
from sklearn.model_selection import KFold
from transformers import AutoModel, AutoTokenizer
import transformers
import wandb
from datetime import datetime as dt
import os
import numpy as np
import tqdm

import logging 
from logging import Logger


@dataclasses.dataclass
class BertConfig:
    
    experiment_name: str
    dataset_dir: str
    
    debug: bool = False

    lr_fc: float = 1e-4
    lr_bert: float = 1e-5
    model_name: str = "microsoft/deberta-v3-large"
    num_context: int = 3
    max_length: int = 512
    batch_size: int = 2
    epochs: int = 10
    iters_to_accumlate: int = 8
    
    freeze_embeddings: bool = True
    freeze_layers: int = 18
    reinitialize_layers: int = 0
    
    assume_completely_retrieved: bool = False
    n_samples: int = None
    

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
    

class MultipleChoiceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: BertConfig,
    ):
        self.df = df
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ret = {}
        for choice in ["A", "B", "C", "D", "E"]:
            context = "context: \n"
            if self.cfg.assume_completely_retrieved and "original_text" in row:
                context += str(row["original_text"])
            else:
                for i in range(self.cfg.num_context):
                    context += str(row[f"searched_wiki_id_{i}"]) + "\n"

            text = str(row["prompt"]) + " [SEP] " + str(row[choice]) + " [SEP] " + context
            text = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.cfg.max_length,
                truncation=True,
                return_tensors="pt",
            )
            ret[choice] = text["input_ids"][0]
        answer_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        label = np.zeros(5)
        label[answer_dict[row["answer"]]] = 1
        ret["label"] = torch.FloatTensor(label)
        ret["id"] = idx
        return ret
            

class BertModel(torch.nn.Module):
    def __init__(
        self,
        config: BertConfig,
        logger: Logger,
    ):
        super().__init__()
        self.config = config
        self.logger = logger
        self.model = AutoModel.from_pretrained(config.model_name, cache_dir="./cache")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir="./cache")
        self.fc = torch.nn.Linear(self.model.config.hidden_size, 1)
        self._model_preprocess()
    
    def _model_preprocess(self):
        ## deberta
        if "microsoft/deberta" in self.config.model_name:
            self.logger.info("freeze embeddings")
            for param in self.model.embeddings.parameters():
                param.requires_grad = False

            self.logger.info(f"freeze {self.config.freeze_layers} layers")
            for layer in self.model.encoder.layer[:self.config.freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
                    
            if self.config.reinitialize_layers > 0:
                self.logger.info(f"reinitialize {self.config.reinitialize_layers} layers")
                for layer in self.model.encoder.layer[-self.config.reinitialize_layers:]:
                    for param in layer.parameters():
                        if isinstance(layer, torch.nn.Linear):
                            param.data.normal_(mean=0.0, std=self.model.config.initializer_range)
                        if isinstance(layer, torch.nn.Linear) and layer.bias is not None:
                            param.bias.data.zero_()
        
    def _forward_prompt(
        self,
        prompt: torch.Tensor,
    ):
        x = self.model(prompt)[0]  # (batch_size, hidden_size, max_length)
        x = x.mean(dim=1)  # (batch_size, hidden_size)
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
        prompt_a = batch["A"].to(device)
        prompt_b = batch["B"].to(device)
        prompt_c = batch["C"].to(device)
        prompt_d = batch["D"].to(device)
        prompt_e = batch["E"].to(device)
        label = batch["label"].to(device)
        
        # training with fp16 and gradient accumulation
        with torch.cuda.amp.autocast():        
            output = model(prompt_a, prompt_b, prompt_c, prompt_d, prompt_e)
            loss = torch.nn.BCEWithLogitsLoss()(output, label)
            loss_meter.update(loss.item(), prompt_a.size(0))
            loss = loss / config.iters_to_accumlate

        scaler.scale(loss).backward()
        if ((i + 1) % model.config.iters_to_accumlate == 0) or (i == (len(loader) - 1)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        tk.set_postfix(
            loss=loss_meter.avg,
            lr=optimizer.param_groups[0]["lr"],
        )
        # wandb にログを送る
        wandb.log({f"train_loss_fold{fold}": loss_meter.avg})
    

def mapk(y_true, y_pred, k=3):
    """
    y_true: (n_samples)
    y_pred: (n_samples, n_classes)
    """
    map_score = 0
    for i in range(len(y_true)):
        y_true_ = y_true[i]
        y_pred_ = y_pred[i]
        score = 0
        for j in range(k):
            if y_pred_[j] == y_true_:
                score += 1 / (j + 1)
                break
        map_score += score
    map_score /= len(y_true)
    return map_score


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
    
    outputs = []
    labels = []
    
    for batch in tqdm.tqdm(loader, desc=f"{name} epoch {epoch}"):
        prompt_a = batch["A"].to(device)
        prompt_b = batch["B"].to(device)
        prompt_c = batch["C"].to(device)
        prompt_d = batch["D"].to(device)
        prompt_e = batch["E"].to(device)
        label = batch["label"].to(device)
        
        with torch.no_grad():
            output = model(prompt_a, prompt_b, prompt_c, prompt_d, prompt_e)
            loss = torch.nn.BCEWithLogitsLoss()(output, label)
        
        outputs.append(torch.sigmoid(output).detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
        loss_meter.update(loss.item(), prompt_a.size(0))
        # wandb にログを送る
        wandb.log({f"{name}_loss_{fold}": loss_meter.avg})
    
    # map@3 を計算する
    outputs = np.concatenate(outputs)
    labels = np.concatenate(labels)

    map3 = mapk(labels.argmax(axis=1), outputs.argsort(axis=1)[:, ::-1], k=3)
    wandb.log({f"{name}_map3_{fold}": map3})
    
    df_valid = pd.DataFrame({
        "pred_A": outputs[:, 0],
        "pred_B": outputs[:, 1],
        "pred_C": outputs[:, 2],
        "pred_D": outputs[:, 3],
        "pred_E": outputs[:, 4],
        "label": labels.argmax(axis=1),
    })
    
    return df_valid, map3
    
    
def main(config: BertConfig):
    try:
        if not torch.cuda.is_available():
            raise ValueError("GPUが使えません")
        output_dir = f"output/{os.path.basename(__file__)}/{dt.now().strftime('%Y%m%d%H%M%S')}_{config.experiment_name}"
        os.makedirs(output_dir, exist_ok=True)
        logger = get_logger(output_dir)
        if os.path.isfile(f"{config.dataset_dir}/searched_index_train.csv"):
            df = pd.read_csv(f"{config.dataset_dir}/searched_index_train.csv")
        elif os.path.isfile(f"{config.dataset_dir}/searched_index_train.parquet"):
            df = pd.read_parquet(f"{config.dataset_dir}/searched_index_train.parquet")
        original_len = len(df)
        df = df[df["answer"].isin(["A", "B", "C", "D", "E"])]
        if config.n_samples is not None:
            df = df.sample(config.n_samples, random_state=42).reset_index(drop=True)
        logger.info(f"original len: {original_len}, after len: {len(df)}")
        
        if config.debug:
            df = df.iloc[:20]
        df_test = pd.read_csv(f"{config.dataset_dir}/searched_index_test.csv")
        
        kfold = KFold(5, random_state=42, shuffle=True)
        
        wandb.init(
            project="llm_science",
            name=config.experiment_name,
            reinit=True,
            tags=["context_pipeline", "stage2"],
        )

        test_dataset = MultipleChoiceDataset(df_test, config)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        for fold, (train_index, valid_index) in enumerate(kfold.split(df)):

            logger.info(f"fold: {fold}")
            model = BertModel(config=config, logger=logger).to("cuda")
            
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
            best_score_val = 0
            best_score_test = 0
            device = "cuda"
            for epoch in range(config.epochs):
                logger.info(f"epoch: {epoch}")
                train_one_epoch(model, train_loader, optimizer, scheduler, device, fold, epoch)
                df_valid_pred, score_val = valid_one_epoch(model, valid_loader, device, fold, epoch, name="valid")
                df_test_pred, score_test = valid_one_epoch(model, test_loader, device, fold, epoch, name="test")
                
                if best_score_val < score_val:
                    logger.info(f"best score(val): {best_score_val} -> {score_val}")
                    best_score_val = score_val
                else:
                    logger.info(f"{score_val} is not better than {best_score_val}(val).")
                    
                # best model を保存する
                if best_score_test < score_test:
                    logger.info(f"best score(test): {best_score_test} -> {score_test}")
                    best_score_test = score_test

                    torch.save(model.state_dict(), f"{output_dir}/model_fold{fold}.pth")
                    pd.concat([df_valid[["prompt", "A", "B", "C", "D", "E", "answer"]], df_valid_pred], axis=1).to_csv(f"{output_dir}/valid_fold{fold}.csv", index=False)
                    pd.concat([df_test[["prompt", "A", "B", "C", "D", "E", "answer"]], df_test_pred], axis=1).to_csv(f"{output_dir}/test_fold{fold}.csv", index=False)
                    wandb.run.summary["best_score_val"] = score_val
                    wandb.run.summary["best_score_test"] = best_score_test
                else:
                    logger.info(f"{score_test} is not better than {best_score_test}(test).")
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
    
    # for freeze_layers in [12, 0]:
    #     for reinit_layers in [0]:
    #         for freeze_embeddings in [True]:
    #             config = BertConfig(
    #                 debug=False,
    #                 epochs=6,
    #                 experiment_name=f"{freeze_layers}layers_freeze_and_freeze_embeddings_{freeze_embeddings}_reinit{reinit_layers}layers",
    #                 freeze_embeddings=freeze_embeddings,
    #                 freeze_layers=freeze_layers,
    #                 reinitialize_layers=reinit_layers,
    #                 dataset_dir="output/context_pipeline/stage1/exp001.py/20230902153542_all-MiniLM-L12-v2_wikiall_targetprompt_and_choice_all_text"
    #             )
    #             main(config=config)

    # for dataset_dir in [
    #     # "output/context_pipeline/stage1/exp001.py/20230903134130_all-MiniLM-L12-v2_wikititle_and_abstract_targetprompt_and_choice_all_text",
    #     "output/context_pipeline/stage1/exp001.py/20230903143444_all-MiniLM-L12-v2_wikititle_and_abstract_targetprompt_and_choice_all_text",
    #     "output/context_pipeline/stage1/exp001.py/20230903143936_multi-qa-mpnet-base-dot-v1_wikititle_and_abstract_targetprompt_and_choice_all_text",
    # ]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=6,
    #         experiment_name=f"use_dataset: {dataset_dir.split('/')[-1]}",
    #         dataset_dir=dataset_dir
    #     )
    #     main(config=config)
    
    # for max_length in [768]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=6,
    #         freeze_layers=18,
    #         assume_completely_retrieved=True,
    #         max_length=max_length,
    #         experiment_name=f"completely_retrieved_maxlength{max_length}",
    #         dataset_dir="output/context_pipeline/stage1/exp001.py/20230902153542_all-MiniLM-L12-v2_wikiall_targetprompt_and_choice_all_text"
    #     )
    #     main(config=config)
    for dataset_dir in [
        "output/context_pipeline/stage1/exp002.py/20230906054359_all-MiniLM-L6-v2_wikionly_text_targetprompt_and_choice_all_text",
    ]:
        config = BertConfig(
            debug=False,
            epochs=6,
            experiment_name=f"use_dataset: {dataset_dir.split('/')[-1]}",
            dataset_dir=dataset_dir,
            n_samples=1500
        )
        main(config=config)
    