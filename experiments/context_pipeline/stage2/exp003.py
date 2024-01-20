
import dataclasses
import torch
import pandas as pd
from sklearn.model_selection import KFold
from transformers import AutoModel, AutoTokenizer, AutoModelForMultipleChoice
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from datasets import Dataset
from typing import Optional, Union

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

    lr: float = 1e-5
    model_name: str = "microsoft/deberta-v3-large"
    num_context: int = 3
    max_length: int = 512
    batch_size: int = 2
    epochs: int = 10
    iters_to_accumlate: int = 8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    freeze_embeddings: bool = True
    freeze_layers: int = 18
    reinitialize_layers: int = 0
    
    assume_completely_retrieved: bool = False
    n_samples: int = None
    steps: int = 100
    
    lora_r: float = 2
    lora_alpha: float = 4
    lora_dropout: float = 0.1
    use_peft: bool = False

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


@dataclasses.dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch


        
def preprocess_df(df, config):
    df["context"] = ""
    for i in range(config.num_context):
        df["context"] += df[f"searched_wiki_id_{i}"].astype(str) + "\n\n"
    
    for col in ["A", "B", "C", "D", "E"]:
        df[col] = df[col].fillna("")
        
    return df[["prompt", "context", "A", "B", "C", "D", "E", "answer"]]


def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}


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
        
        df = preprocess_df(df, config)
        df_test = preprocess_df(df_test, config)
        
        if len(df) > 30000:
            kfold = KFold(100, random_state=42, shuffle=True)
        else:
            kfold = KFold(5, random_state=42, shuffle=True)
        
        wandb.init(
            project="llm_science",
            name=config.experiment_name,
            reinit=True,
            tags=["context_pipeline", "stage2"],
        )

        tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir="./cache")
        option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
        index_to_option = {v: k for k,v in option_to_index.items()}
        def preprocess(example):
            first_sentence = [ "[CLS] " + example['context'] ] * 5
            second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
            tokenized_example = tokenizer(first_sentence, second_sentences, truncation="only_first", 
                                          max_length=config.max_length, add_special_tokens=False)
            tokenized_example['label'] = option_to_index[example['answer']]
            
            return tokenized_example
        
        test_dataset = Dataset.from_pandas(df_test)
        tokenized_test_dataset = test_dataset.map(preprocess, remove_columns=["prompt", "context", "A", "B", "C", "D", "E"])
        
        for fold, (train_index, valid_index) in enumerate(kfold.split(df)):

            logger.info(f"fold: {fold}")
            model = AutoModelForMultipleChoice.from_pretrained(config.model_name, cache_dir="./cache")
            
            if config.use_peft:
                logger.info('We are using PEFT.')
                from peft import LoraConfig, get_peft_model, TaskType
                peft_config = LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha, 
                    task_type=TaskType.SEQ_CLS, 
                    lora_dropout=config.lora_dropout, 
                    bias="none", 
                    inference_mode=False, 
                    target_modules=["query_proj", "value_proj", "key_proj", "dense"],
                    modules_to_save=['classifier','pooler'],
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            if "deberta" in config.model_name:
                if config.freeze_embeddings:
                    logger.info("freeze embeddings")
                    for param in model.deberta.embeddings.parameters():
                        param.requires_grad = False

                logger.info(f"freeze {config.freeze_layers} layers")
                for layer in model.deberta.encoder.layer[:config.freeze_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False
                
                if config.reinitialize_layers > 0:            
                    logger.info(f"reinitialize {config.reinitialize_layers} layers")
                    for layer in model.deberta.encoder.layer[-config.reinitialize_layers:]:
                        for param in layer.parameters():
                            if isinstance(layer, torch.nn.Linear):
                                param.data.normal_(mean=0.0, std=model.config.initializer_range)
                            if isinstance(layer, torch.nn.Linear) and layer.bias is not None:
                                param.bias.data.zero_()
            if "roberta" in config.model_name:
                if config.freeze_embeddings:
                    logger.info("freeze embeddings")
                    for param in model.roberta.embeddings.parameters():
                        param.requires_grad = False

                logger.info(f"freeze {config.freeze_layers} layers")
                for layer in model.roberta.encoder.layer[:config.freeze_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False
                
                if config.reinitialize_layers > 0:            
                    logger.info(f"reinitialize {config.reinitialize_layers} layers")
                    for layer in model.roberta.encoder.layer[-config.reinitialize_layers:]:
                        for param in layer.parameters():
                            if isinstance(layer, torch.nn.Linear):
                                param.data.normal_(mean=0.0, std=model.config.initializer_range)
                            if isinstance(layer, torch.nn.Linear) and layer.bias is not None:
                                param.bias.data.zero_()
            
            df_train = df.iloc[train_index].reset_index(drop=True)
            df_valid = df.iloc[valid_index].reset_index(drop=True)
            # DataLoader の作成
            train_dataset = Dataset.from_pandas(df_train)
            tokenized_train_dataset = train_dataset.map(preprocess, remove_columns=["prompt", "context", "A", "B", "C", "D", "E"])
            valid_dataset = Dataset.from_pandas(df_valid)
            tokenized_valid_dataset = valid_dataset.map(preprocess, remove_columns=["prompt", "context", "A", "B", "C", "D", "E"])

            training_args = TrainingArguments(
                warmup_ratio=config.warmup_ratio, 
                learning_rate=config.lr,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                num_train_epochs=config.epochs,
                report_to='wandb',
                output_dir=output_dir,
                overwrite_output_dir=True,
                fp16=True,
                gradient_accumulation_steps=config.iters_to_accumlate,
                logging_steps=25,
                evaluation_strategy='steps',
                eval_steps=config.steps,
                save_strategy="steps",
                save_steps=config.steps,
                load_best_model_at_end=True,
                metric_for_best_model='test_map@3',
                lr_scheduler_type='cosine',
                weight_decay=config.weight_decay,
                save_total_limit=1,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
                data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
                train_dataset=tokenized_train_dataset,
                eval_dataset={
                    "eval": tokenized_valid_dataset,
                    "test": tokenized_test_dataset,
                },
                compute_metrics=compute_metrics,
                #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
            )

            trainer.train()
            trainer.save_model(f"{output_dir}/fold{fold}")
            
            test_predictions = trainer.predict(tokenized_test_dataset).predictions
            for i, col in enumerate(["A", "B", "C", "D", "E"]):
                df_test[f"fold{fold}_{col}"] = test_predictions[:, i]
            
            score = map_at_3(test_predictions, df_test["answer"].map(option_to_index))
            logger.info(f"map@3: {score}")
            wandb.log({"test/map@3": score})
            
            df_test.to_csv(f"{output_dir}/test_predictions.csv", index=False)
            break
    except Exception as e:
        logger.exception(e)
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

    # for dataset_dir in [
    #     "output/context_pipeline/stage1/exp002.py/20230906054359_all-MiniLM-L6-v2_wikionly_text_targetprompt_and_choice_all_text",
    # ]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=6,
    #         batch_size=2,
    #         experiment_name=f"use_dataset: {dataset_dir.split('/')[-1]}",
    #         dataset_dir=dataset_dir,
    #         max_length=256,
    #         n_samples=1500,
    #         num_context=1,
    #     )
    #     main(config=config)
    

    # for lr in [5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 4e-5]:    
    #     config = BertConfig(
    #         debug=False,
    #         epochs=6,
    #         batch_size=2,
    #         experiment_name=f"lr = {lr}",
    #         lr=lr,
    #         dataset_dir=dataset_dir,
    #         max_length=256,
    #         n_samples=1500,
    #         num_context=1,
    #     )
    #     main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=6,
    #     batch_size=2,
    #     experiment_name=f"max_length=512",
    #     dataset_dir=dataset_dir,
    #     max_length=512,
    #     n_samples=1500,
    #     num_context=1,
    # )
    # main(config=config)
    
    # for n_samples in [3000, 6000, 12000]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=6,
    #         batch_size=2,
    #         experiment_name=f"max_length=512",
    #         dataset_dir=dataset_dir,
    #         max_length=256,
    #         n_samples=n_samples,
    #         num_context=1,
    #     )
    #     main(config=config)
    
    # for freeze_layers in [0, 6, 12, 15, 18, 21]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=12,
    #         batch_size=1,
    #         experiment_name=f"freeze_layers_{freeze_layers}",
    #         dataset_dir=dataset_dir,
    #         freeze_layers=freeze_layers,
    #         max_length=512,
    #         n_samples=1500,
    #         num_context=1,
    #     )
    #     main(config=config)
    
    # for lr in [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 4e-5]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=12,
    #         batch_size=1,
    #         lr=lr,
    #         experiment_name=f"freeze_layers_0_maxlength512_lr{lr}",
    #         dataset_dir=dataset_dir,
    #         freeze_layers=0,
    #         freeze_embeddings=False,
    #         max_length=512,
    #         n_samples=1500,
    #         num_context=1,
    #     )
    #     main(config=config)

    # for num_context in [3, 5]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=12,
    #         batch_size=1,
    #         num_context=num_context,
    #         experiment_name=f"freeze_layers_0_maxlength512_num_context{num_context}",
    #         dataset_dir=dataset_dir,
    #         freeze_layers=0,
    #         freeze_embeddings=False,
    #         max_length=512,
    #         n_samples=1500,
    #     )
    #     main(config=config)
    
    # for weight_decay in [1e-1, 3e-2, 3e-3, 1e-3]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=12,
    #         batch_size=1,
    #         weight_decay=weight_decay,
    #         experiment_name=f"freeze_layers_0_maxlength512_weight_decay{weight_decay}",
    #         dataset_dir=dataset_dir,
    #         freeze_layers=0,
    #         freeze_embeddings=False,
    #         max_length=512,
    #         n_samples=1500,
    #         num_context=1,
    #     )
    #     main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=1,
    #     freeze_layers=0,
    #     freeze_embeddings=False,
    #     lr=4e-5,
    #     experiment_name=f"max_length=512 freeze_layers=0",
    #     dataset_dir=dataset_dir,
    #     max_length=512,
    #     num_context=1,
    # )
    # main(config=config)
    
    # for lr in [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 4e-5]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=6,
    #         batch_size=2,
    #         lr=lr,
    #         experiment_name=f"freeze_layers_18_maxlength512_lr{lr}",
    #         dataset_dir=dataset_dir,
    #         freeze_layers=18,
    #         max_length=512,
    #         n_samples=1500,
    #         num_context=3,
    #     )
    #     main(config=config)
        
    # for lr in [2e-5, 4e-5, 8e-5]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=6,
    #         batch_size=2,
    #         lr=lr,
    #         experiment_name=f"freeze_layers_21_maxlength512_lr{lr}",
    #         dataset_dir=dataset_dir,
    #         freeze_layers=21,
    #         max_length=512,
    #         n_samples=1500,
    #         num_context=3,
    #     )
    #     main(config=config)
    
    # for n_samples in [3000, 6000, 12000]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=6,
    #         batch_size=2,
    #         lr=4e-5,
    #         experiment_name=f"freeze_layers_21_maxlength512_lr4e-5",
    #         dataset_dir=dataset_dir,
    #         freeze_layers=21,
    #         max_length=512,
    #         n_samples=n_samples,
    #         num_context=3,
    #     )
    #     main(config=config)
    # for iter_to_accumulate in [16, 32, 64]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=6,
    #         batch_size=2,
    #         lr=4e-5,
    #         experiment_name=f"freeze_layers_21_maxlength512_lr4e-5_iter_to_accumulate{iter_to_accumulate}",
    #         dataset_dir=dataset_dir,
    #         freeze_layers=21,
    #         max_length=512,
    #         n_samples=1500,
    #         iters_to_accumlate=iter_to_accumulate,
    #         num_context=3,
    #     )
    #     main(config=config)
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=6,
    #     batch_size=1,
    #     lr=2e-6,
    #     experiment_name=f"freeze_layers_0_maxlength256_all_data_lr2e-6",
    #     dataset_dir=dataset_dir,
    #     freeze_layers=0,
    #     freeze_embeddings=False,
    #     steps=600,
    #     max_length=256,
    #     num_context=3,
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=6,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze_layers_18_maxlength256_all_data_lr5e-6",
    #     dataset_dir=dataset_dir,
    #     freeze_layers=18,
    #     max_length=256,
    #     steps=600,
    #     num_context=3,
    # )
    # main(config=config)
       
    # for lr in [1e-6, 2e-6, 5e-6]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=30,
    #         batch_size=2,
    #         lr=lr,
    #         experiment_name=f"freeze_layers_18_maxlength512_lr{lr}",
    #         dataset_dir=dataset_dir,
    #         freeze_layers=18,
    #         max_length=512,
    #         n_samples=1500,
    #         num_context=3,
    #     )
    #     main(config=config)
    
    # for lr in [4e-5]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=6,
    #         batch_size=2,
    #         lr=lr,
    #         experiment_name=f"freeze_layers_18_maxlength512_lr{lr}",
    #         dataset_dir="output/context_pipeline/stage1/exp003.py/20230909002518_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride50_all",
    #         freeze_layers=18,
    #         max_length=512,
    #         n_samples=1500,
    #         num_context=3,
    #     )
    #     main(config=config)
        
    # for lr in [1e-6, 2e-6]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=30,
    #         batch_size=1,
    #         lr=lr,
    #         experiment_name=f"freeze_layers_0_maxlength512_lr{lr}",
    #         dataset_dir="output/context_pipeline/stage1/exp002.py/20230906054359_all-MiniLM-L6-v2_wikionly_text_targetprompt_and_choice_all_text",
    #         freeze_layers=0,
    #         freeze_embeddings=False,
    #         max_length=512,
    #         n_samples=1500,
    #         num_context=3,
    #     )
    #     main(config=config)
        
    # for lr in [1e-6, 2e-6, 5e-6]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=lr,
    #         experiment_name=f"freeze18_maxlength256_lr{lr}_alldata",
    #         dataset_dir="output/context_pipeline/stage1/exp003.py/20230909002518_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride50_all",
    #         freeze_layers=18,
    #         max_length=256,
    #         num_context=1,
    #     )
    #     main(config=config)
    
    # for iters_to_accumulate in [16, 32, 64]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=5e-6,
    #         iters_to_accumlate=iters_to_accumulate,
    #         experiment_name=f"freeze18_maxlength256_accumulate{iters_to_accumulate}_alldata",
    #         dataset_dir="output/context_pipeline/stage1/exp003.py/20230909002518_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride50_all",
    #         freeze_layers=18,
    #         max_length=256,
    #         num_context=1,
    #     )
    #     main(config=config)
    
    # for lr in [1e-5, 2e-5, 4e-5]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=lr,
    #         experiment_name=f"freeze18_maxlength256_lr{lr}_alldata",
    #         dataset_dir="output/context_pipeline/stage1/exp003.py/20230909002518_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride50_all",
    #         freeze_layers=18,
    #         max_length=256,
    #         num_context=1,
    #     )
    #     main(config=config)
    
    # for iters_to_accumulate in [4, 16, 32, 64]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=5e-6 * iters_to_accumulate / 8,
    #         iters_to_accumlate=iters_to_accumulate,
    #         experiment_name=f"freeze18_maxlength256_accumulate{iters_to_accumulate}_alldata",
    #         dataset_dir="output/context_pipeline/stage1/exp003.py/20230909002518_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride50_all",
    #         freeze_layers=18,
    #         max_length=256,
    #         num_context=1,
    #     )
    #     main(config=config)
    # for freeze_layers in [0, 12]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=2e-5,
    #         experiment_name=f"freeze{freeze_layers}_maxlength256_lr2e-5_alldata",
    #         dataset_dir="output/context_pipeline/stage1/exp003.py/20230909002518_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride50_all",
    #         freeze_layers=freeze_layers,
    #         max_length=256,
    #         num_context=1,
    #     )
    #     main(config=config)

    # for freeze_layers in [18]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=6,
    #         batch_size=2,
    #         lr=2e-5,
    #         experiment_name=f"freeze{freeze_layers}_maxlength256_lr2e-5_20230906gpt",
    #         dataset_dir="output/context_pipeline/stage1/exp003.py/20230910082305_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride50_all",
    #         freeze_layers=freeze_layers,
    #         max_length=256,
    #         num_context=1,
    #     )
    #     main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=2e-4,
    #     experiment_name=f"freeze0_maxlength256_lr2e-4_all_lora",
    #     dataset_dir="output/context_pipeline/stage1/exp003.py/20230909002518_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride50_all",
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=1,
    #     use_peft=True,
    # )
    # main(config=config)
        
    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength256_lr2e-5_maxlength100",
    #     dataset_dir="output/context_pipeline/stage1/exp004.py/20230910131328_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride50_all",
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    # )
    # main(config=config) 
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength256_lr2e-5_maxlength200",
    #     dataset_dir="output/context_pipeline/stage1/exp004.py/20230910180056_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length200_stride100_all",
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=1,
    # )
    # main(config=config) 
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=4e-4,
    #     experiment_name=f"freeze0_maxlength256_lr4e-4_all_lora",
    #     dataset_dir="output/context_pipeline/stage1/exp003.py/20230909002518_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride50_all",
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=1,
    #     use_peft=True,
    # )
    # main(config=config)

    
    # for model_name in ["roberta-large", "OpenAssistant/reward-model-deberta-v3-large-v2"]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=5e-6,
    #         experiment_name=f"freeze0_maxlength256_lr5e-6_{model_name}",
    #         model_name=model_name,
    #         dataset_dir="output/context_pipeline/stage1/exp003.py/20230909002518_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride50_all",
    #         freeze_layers=0,
    #         max_length=256,
    #         num_context=1,
    #     )
    #     main(config=config) 
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     reinitialize_layers=2,
    #     experiment_name=f"freeze0_maxlength256_lr5e-6_reinit2layers",
    #     dataset_dir="output/context_pipeline/stage1/exp003.py/20230909002518_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride50_all",
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=1,
    # )
    # main(config=config) 
    
    for model_name in ["OpenAssistant/reward-model-deberta-v3-large-v2"]:
        config = BertConfig(
            debug=False,
            epochs=2,
            batch_size=2,
            lr=5e-6,
            experiment_name=f"freeze0_maxlength256_lr5e-6_{model_name}_200-stride100",
            model_name=model_name,
            dataset_dir="output/context_pipeline/stage1/exp006.py/20230914032500_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length200_stride100_all",
            freeze_layers=0,
            max_length=256,
            num_context=1,
        )
        main(config=config) 
