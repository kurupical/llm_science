
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
from typing import List
from logging import Logger



@dataclasses.dataclass
class BertConfig:
    
    experiment_name: str
    dataset_dir: str
    
    debug: bool = False

    lr: float = 1e-5
    model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"
    num_context: int = 3
    max_length: int = 512
    batch_size: int = 2
    epochs: int = 10
    iters_to_accumlate: int = 8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    freeze_embeddings: bool = True
    freeze_layers: int = 18
    reinitialize_layers: int = 0
    
    assume_completely_retrieved: bool = False
    n_samples: int = None
    steps: int = 500
     
    lora_r: float = 2
    lora_alpha: float = 4
    lora_dropout: float = 0.1
    use_peft: bool = False
    metric_for_best_model: str = "eval_eval_map@3"
    greater_is_better: bool = True
    
    data_source: List[str] = None
    data_source_without: List[str] = None
    lr_scheduler_type: str = "cosine"
    
    optim: str = "adamw_hf"
    sep_token: str = "\n\n"

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
        df["context"] += df[f"searched_wiki_id_{i}"].astype(str) + config.sep_token
    
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
        output_dir = f"output/stage2/{os.path.basename(__file__)}/{dt.now().strftime('%Y%m%d%H%M%S')}_{config.experiment_name}"
        os.makedirs(output_dir, exist_ok=True)
        logger = get_logger(output_dir)
        if os.path.isfile(f"{config.dataset_dir}/train.parquet"):
            df_train = pd.read_parquet(f"{config.dataset_dir}/train.parquet")
        elif os.path.isfile(f"{config.dataset_dir}/train.feather"):
            df_train = pd.read_feather(f"{config.dataset_dir}/train.feather")
        original_len = len(df_train)
        if config.data_source is not None:
            logger.info(f"data source filter: {config.data_source}")
            df_train = df_train[df_train["source"].isin(config.data_source)]
        if config.data_source_without is not None:
            logger.info(f"data source filter: {config.data_source_without}")
            df_train = df_train[~df_train["source"].isin(config.data_source_without)]
        if os.path.isfile(f"{config.dataset_dir}/valid.parquet"):
            df_valid = pd.read_parquet(f"{config.dataset_dir}/valid.parquet")
        elif os.path.isfile(f"{config.dataset_dir}/valid.feather"):
            df_valid = pd.read_feather(f"{config.dataset_dir}/valid.feather")
        df_train = df_train[df_train["answer"].isin(["A", "B", "C", "D", "E"])]
        df_valid = df_valid[df_valid["answer"].isin(["A", "B", "C", "D", "E"])]
        if config.n_samples is not None:
            df_train = df_train.sample(config.n_samples, random_state=42).reset_index(drop=True)
        logger.info(f"original len: {original_len}, after len: {len(df_train)}")
        logger.info(f"data_source: {df_train['source'].value_counts()}")
        
        if config.debug:
            df_train = df_train.iloc[:20]
        if os.path.isfile(f"{config.dataset_dir}/test.parquet"):
            df_test = pd.read_parquet(f"{config.dataset_dir}/test.parquet")
        elif os.path.isfile(f"{config.dataset_dir}/test.feather"):
            df_test = pd.read_feather(f"{config.dataset_dir}/test.feather")
        df_train = preprocess_df(df_train, config)
        if config.num_context != 0:
            df_train = df_train[df_train["context"].str.len() > 30] # トークン数短すぎると怒られるので
        # for col in ["prompt", "A", "B", "C", "D", "E"]:
        #     df_train = df_train[df_train[col].str.len() < 400]
        df_valid = preprocess_df(df_valid, config)
        
        df_test = preprocess_df(df_test, config)
        
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
            try:
                if config.num_context == 0:
                    first_sentence = [ "[CLS] " + example['prompt'] ] * 5
                    second_sentences = [" #### " + example[option] + " [SEP]" for option in 'ABCDE']
                    tokenized_example = tokenizer(first_sentence, second_sentences, truncation=True, 
                                                max_length=config.max_length, add_special_tokens=False)
                    tokenized_example['label'] = option_to_index[example['answer']]
                else:            
                    first_sentence = [ "[CLS] " + example['context'] ] * 5
                    second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
                    tokenized_example = tokenizer(first_sentence, second_sentences, truncation=True, 
                                                max_length=config.max_length, add_special_tokens=False)
                    tokenized_example['label'] = option_to_index[example['answer']]
                return tokenized_example
            except Exception as e:
                print(e)
                print(first_sentence)
                print(second_sentences)
                return None
        
        test_dataset = Dataset.from_pandas(df_test)
        tokenized_test_dataset = test_dataset.map(preprocess, remove_columns=["prompt", "context", "A", "B", "C", "D", "E"])
        
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
        
        fold = 0
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
            optim=config.optim,
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
            metric_for_best_model=config.metric_for_best_model,
            greater_is_better=config.greater_is_better,
            lr_scheduler_type=config.lr_scheduler_type,
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
        
        valid_predictions = trainer.predict(tokenized_valid_dataset).predictions
        for i, col in enumerate(["A", "B", "C", "D", "E"]):
            df_valid[f"fold{fold}_{col}"] = valid_predictions[:, i]
        df_valid.to_csv(f"{output_dir}/valid_predictions.csv", index=False)
        
        test_predictions = trainer.predict(tokenized_test_dataset).predictions
        for i, col in enumerate(["A", "B", "C", "D", "E"]):
            df_test[f"fold{fold}_{col}"] = test_predictions[:, i]
        df_test.to_csv(f"{output_dir}/test_predictions.csv", index=False)
        
        score = map_at_3(test_predictions, df_test["answer"].map(option_to_index))
        logger.info(f"map@3: {score}")
        wandb.log({"test/map@3": score})
        
    except Exception as e:
        logger.exception(e)
    finally:
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
        wandb.finish()



if __name__ == "__main__":
    
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength256_context3_lr5e-6_100-stride75_merge",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230917055045_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    #     n_samples=1500,
    #     data_source=["3", "10", "2", "6", "5", "4", "8", "9", "7", "11", "1"],
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength256_context3_lr5e-6_100-stride75_merge",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230917055045_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    #     n_samples=1500,
    #     data_source=[
    #         'additional_data/ScienceQA/train.parquet',
    #         'additional_data/ScienceQA/test.parquet',
    #         'additional_data/ScienceQA/val.parquet'
    #     ],
    # )
    # main(config=config)
    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength256_context3_lr5e-6_100-stride75_merge",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230917055045_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    #     n_samples=1500,
    #     data_source=[
    #         'additional_data/OpenBookQA-V1-Sep2018/OpenBookQA-V1-Sep2018/Data/Main/train.tsv',
    #         'additional_data/OpenBookQA-V1-Sep2018/OpenBookQA-V1-Sep2018/Data/Main/test.tsv',
    #         'additional_data/OpenBookQA-V1-Sep2018/OpenBookQA-V1-Sep2018/Data/Main/dev.tsv'
    #     ],
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength256_context3_lr5e-6_100-stride75_merge",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230917055045_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    #     n_samples=1500,
    #     data_source=[
    #         'additional_data/MMLU/test.csv', 
    #         'additional_data/MMLU/train.csv',
    #         'additional_data/MMLU/valid.csv'
    #     ],
    # )
    
    # main(config=config)

    # for iters_to_accumulate in [32, 64]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=3,
    #         batch_size=2,
    #         lr=5e-6,
    #         iters_to_accumlate=iters_to_accumulate,
    #         experiment_name=f"maxlength256_lr5e-6_100-stride75_iters_to_accumulate{iters_to_accumulate}_3epochs",
    #         dataset_dir="data/train_parquet/original",
    #         freeze_layers=0,
    #         max_length=256,
    #         num_context=2,
    #         steps=1000*8//iters_to_accumulate,
    #     )
    #     main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=1,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength512_context5_lr5e-6_100-stride75",
    #     dataset_dir="data/train_parquet/merge",
    #     freeze_layers=0,
    #     max_length=512,
    #     iters_to_accumlate=16,
    #     num_context=5,
    # )
    # main(config=config)
        
    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength256_context3_lr5e-6_100-stride75_merge_newval_baseline",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230917055045_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    #     data_source=["3", "10", "2", "6", "5", "4", "8", "9", "7", "11", "1"],
    # )
    # main(config=config)
    
    # for data_source in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
    #                     'additional_data/MMLU/train.csv',
    #                     'additional_data/OpenBookQA-V1-Sep2018/OpenBookQA-V1-Sep2018/Data/Main/train.tsv',
    #                     'additional_data/ScienceQA/train.parquet',
    #                     ]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=4,
    #         batch_size=2,
    #         lr=5e-6,
    #         experiment_name=f"freeze0_maxlength256_context3_lr5e-6_100-stride75_only_{data_source}",
    #         dataset_dir="output/context_pipeline/stage1/exp007.py/20230917055045_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #         freeze_layers=0,
    #         max_length=384,
    #         num_context=3,
    #         n_samples=1000,
    #         steps=60,
    #         data_source=[data_source],
    #     )
    #     main(config=config)
        
    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength256_context3_lr5e-6_100-stride75_merge_newval_additionaldata",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230917055045_multi-qa-mpnet-base-dot-v1_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
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
    #     experiment_name=f"freeze0_maxlength384_context3_lr5e-6_100-stride75_without_mygpt",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230918111644_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=0,
    #     max_length=384,
    #     num_context=3,
    #     data_source_without=["20230914125028_20230911_gpt3.5_generate4"]
    # )
    # main(config=config)
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength384_context3_lr5e-6_100-stride75_all",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230918111644_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=0,
    #     max_length=384,
    #     num_context=3,
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength384_context3_lr5e-6_100-stride75_only_gpt_without_mine",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230918111644_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=0,
    #     max_length=384,
    #     num_context=3,
    #     data_source=["1", "2", "3", "4", "5", "6", "7", "8"]
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=1,
    #     iters_to_accumlate=16,
    #     lr=2e-5,
    #     experiment_name=f"freeze6_maxlength1024_context9_lr2e-5_100-stride75_deberta-base",
    #     model_name="OpenAssistant/reward-model-deberta-v3-base",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230918111644_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=6,
    #     max_length=1024,
    #     num_context=9,
    # )
    # main(config=config)
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength384_context3_lr5e-6_100-stride75_with_constant_scheduler",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230918222000_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=0,
    #     max_length=384,
    #     num_context=3,
    #     lr_scheduler_type="constant",
    #     data_source_without=["20230914125028_20230911_gpt3.5_generate4"]
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength384_context3_lr5e-6_100-stride75_all_dataset",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230918222000_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=0,
    #     max_length=384,
    #     num_context=3,
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"unfreeze_maxlength384_context3_lr5e-6_100-stride75",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230918222000_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=0,
    #     max_length=384,
    #     num_context=3,
    #     freeze_embeddings=False,
    #     data_source_without=["20230914125028_20230911_gpt3.5_generate4"]
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength384_context3_lr5e-6_100-stride75_wd0.1",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230918222000_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=0,
    #     max_length=384,
    #     num_context=3,
    #     weight_decay=0.1,
    #     data_source_without=["20230914125028_20230911_gpt3.5_generate4"]
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=1,
    #     iters_to_accumlate=16,
    #     lr=2e-5,
    #     experiment_name=f"freeze18_maxlength768_context7_lr2e-5_100-stride75_only_gpt_with_mine",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230918111644_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=18,
    #     max_length=768,
    #     num_context=7,
    #     data_source=["1", "2", "3", "4", "5", "6", "7", "8", "20230914125028_20230911_gpt3.5_generate4"]
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength384_context3_lr5e-6_100-stride75_wd0.1",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230918222000_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=0,
    #     max_length=384,
    #     num_context=3,
    #     weight_decay=0.1,
    #     data_source_without=["20230914125028_20230911_gpt3.5_generate4"]
    # )
    # main(config=config)

    # for model_name in [
    #     "sileod/deberta-v3-large-tasksource-rlhf-reward-model",
    #     "allenai/longformer-scico",
    # ]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=5e-6,
    #         model_name=model_name,
    #         experiment_name=f"freeze0_maxlength384_context3_lr2e-5_100-stride75_only_gpt_model{model_name}",
    #         dataset_dir="output/context_pipeline/stage1/exp007.py/20230918222000_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #         freeze_layers=0,
    #         max_length=384,
    #         num_context=3,
    #         data_source=["1", "2", "3", "4", "5", "6", "7", "8"]
    #     )
    #     main(config=config)

    # for lr in [2e-6, 5e-6]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=lr,
    #         optim="adafactor",
    #         experiment_name=f"freeze0_maxlength384_num_content3_lr{lr}_100-stride75_only_gpt_adafactor",
    #         dataset_dir="output/context_pipeline/stage1/exp007.py/20230918222000_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #         freeze_layers=0,
    #         max_length=384,
    #         num_context=3,
    #         data_source=["1", "2", "3", "4", "5", "6", "7", "8"]
    #     )
    #     main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=1,
    #     iters_to_accumlate=16,
    #     lr=5e-6,
    #     experiment_name=f"freeze0_maxlength384_context3_lr2e-5_100-stride75_only_gpt_iter16_bs1",
    #     dataset_dir="output/context_pipeline/stage1/exp007.py/20230918222000_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length100_stride75_all",
    #     freeze_layers=0,
    #     max_length=384,
    #     num_context=3,
    #     data_source=["1", "2", "3", "4", "5", "6", "7", "8"]
    # )
    # main(config=config)

    # for max_length, num_content in zip([256, 384, 512], [2, 3, 5]):
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         iters_to_accumlate=16,
    #         batch_size=1,
    #         lr=5e-6,
    #         steps=1000,
    #         experiment_name=f"new_data_all300val_maxlen{max_length}",
    #         dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230922162941_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #         freeze_layers=0,
    #         max_length=max_length,
    #         num_context=num_content,
    #     )
    #     main(config=config)

    # for max_length, num_content in zip([512], [4]):
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         iters_to_accumlate=16,
    #         batch_size=1,
    #         lr=5e-6,
    #         steps=1000,
    #         experiment_name=f"new_data_all300val_maxlen{max_length}",
    #         dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230922162941_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #         freeze_layers=0,
    #         max_length=max_length,
    #         num_context=num_content,
    #     )
    #     main(config=config)
    
    # for lr in [5e-6, 1e-6]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         iters_to_accumlate=1,
    #         batch_size=2,
    #         lr=lr,
    #         steps=1000*16,
    #         experiment_name=f"new_data_all300val_maxlen386_accumulate1_lr{lr}",
    #         dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230922162941_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #         freeze_layers=0,
    #         max_length=386,
    #         num_context=3,
    #     )
    #     main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     iters_to_accumlate=16,
    #     batch_size=1,
    #     lr=5e-6,
    #     steps=1000,
    #     experiment_name=f"new_data_all300val_maxlen386_[SEP]",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230922162941_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=386,
    #     num_context=3,
    #     sep_token="[SEP]"
    # )
    # main(config=config)
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=4,
    #     iters_to_accumlate=4,
    #     lr=5e-6,
    #     steps=1000,
    #     experiment_name=f"new_data_all300val_maxlen128_only_prompt",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230925104920_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=128,
    #     num_context=0,
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=5e-6,
    #     steps=1000,
    #     experiment_name=f"new_data_all300val_maxlen256_4epochs",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230925104920_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    # )
    # main(config=config)
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=2e-6,
    #     steps=1000,
    #     experiment_name=f"new_data_all300val_maxlen256_4epochs_lr2e-6",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230925104920_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    # )
    # main(config=config)
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=4,
    #     iters_to_accumlate=4,
    #     lr=2e-5,
    #     steps=1000,
    #     experiment_name=f"new_data_all300val_maxlen256_freeze18_bs4",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230922162941_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=18,
    #     max_length=256,
    #     num_context=2,
    # )
    # main(config=config)

    # for model_name in [
    #     "OpenAssistant/reward-model-deberta-v3-base",
    #     "microsoft/deberta-v3-large",
    # ]:            
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=5e-6,
    #         steps=1000,
    #         experiment_name=f"new_data_all300val_maxlen256_bs2_{model_name}",
    #         dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230922162941_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #         freeze_layers=0,
    #         max_length=256,
    #         model_name=model_name,
    #         num_context=2,
    #     )
    #     main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=4,
    #     iters_to_accumlate=4,
    #     lr=5e-6,
    #     steps=1000,
    #     use_peft=True,
    #     experiment_name=f"new_data_all300val_maxlen256_lora_bs4",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230922162941_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
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
    #     steps=1000,
    #     experiment_name=f"new_data_all300val_maxlen256_bs2_without_mmlu",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230922162941_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    #     data_source_without=["additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv"]
    # )
    # main(config=config)
        
    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     steps=1000,
    #     experiment_name=f"new_data_all300val_maxlen256_bs2_addgpt6",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230927110701_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all/concat",  # ここに作ったデータ入れる
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
    #     steps=1000,
    #     experiment_name=f"new_data_all300val_maxlen256_bs2",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230922162941_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
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
    #     warmup_ratio=0.01,
    #     steps=1000,
    #     experiment_name=f"new_data_all300val_maxlen256_bs2_warm0.01",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230922162941_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    # )
    # main(config=config)

    # # イイ結果のもので long epochs 回す.
    # config = BertConfig(
    #     debug=False,
    #     epochs=3,
    #     batch_size=2,
    #     lr=2e-6,
    #     steps=1000,
    #     experiment_name=f"new_data_all300val_maxlen256_bs2_lr2e-6_3epochs",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230922162941_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
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
    #     steps=1000,
    #     experiment_name=f"new_data_all300val_maxlen256_bs2_without_additional_3",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230922162941_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "additional_data/ScienceQA/train.parquet", "additional_data/ScienceQA/test.parquet", 
    #         "additional_data/OpenBookQA-V1-Sep2018/OpenBookQA-V1-Sep2018/Data/Main/train.tsv",
    #     ]
    # )
    # main(config=config)
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     steps=1000,
    #     experiment_name=f"new_data_all300val_maxlen256_bs2_only_gpt",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230922162941_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    #     data_source=[
    #         "1", "2", "3", "4", "5", "6", "7", "8"
    #     ]
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     steps=1000,
    #     experiment_name=f"new_data_all300val_maxlen256_bs2_retrieval_nlists1",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20230928215531_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "20230924083304_gpt5",
    #     ]
    # )
    # main(config=config)
    
    # for max_length in [384, 256]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=5e-6,
    #         steps=1000,
    #         experiment_name=f"l150_s6_bge_small_maxlen{max_length}_bs2",
    #         dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231001001643_bge-small-en-v1.5_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length150_stride_sentence6_drop_categoryTrue_all", 
    #         freeze_layers=0,
    #         max_length=max_length,
    #         num_context=2,
    #         data_source_without=[
    #             "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #             "20230924083304_gpt5",
    #         ]
    #     )
    #     main(config=config)

    # for model_name in [
    #     "OpenAssistant/reward-model-deberta-v3-large-v2",
    #     "microsoft/deberta-v3-large",
    # ]:            
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=5e-6,
    #         steps=1000,
    #         experiment_name=f"gtesmall_l120s4_maxlen256_bs2_{model_name}",
    #         dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231001150531_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #         freeze_layers=0,
    #         max_length=256,
    #         model_name=model_name,
    #         num_context=2,
    #         data_source_without=[
    #             "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #             "20230924083304_gpt5",
    #         ]
    #     )
    #     main(config=config)
        
    # for grad_norm in [0.1, 10]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=5e-6,
    #         steps=1000,
    #         max_grad_norm=grad_norm,
    #         experiment_name=f"gtesmall_l120s4_maxlen256_bs2_gradnorm{grad_norm}",
    #         dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231001150531_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #         freeze_layers=0,
    #         max_length=256,
    #         num_context=2,
    #         data_source_without=[
    #             "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #             "20230924083304_gpt5",
    #         ]
    #     )
    #     main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     experiment_name=f"gtesmall_l120s4_maxlen256_bs2_warmup_ratio0.01",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231001150531_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "20230924083304_gpt5",
    #     ]
    # )
    # main(config=config)

    # for model_name in [
    #     # "OpenAssistant/reward-model-deberta-v3-large-v2",
    #     "microsoft/deberta-v3-large",
    # ]:            
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=5e-6,
    #         steps=1000,
    #         experiment_name=f"gtesmall_l90s3_maxlen256_bs2_{model_name}",
    #         dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231003051044_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #         freeze_layers=0,
    #         max_length=256,
    #         model_name=model_name,
    #         num_context=3,
    #         data_source_without=[
    #             "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #             "20230924083304_gpt5",
    #         ]
    #     )
    #     main(config=config)
    # for model_name in [
    #     "OpenAssistant/reward-model-deberta-v3-large-v2",
    #     # "microsoft/deberta-v3-large",
    # ]:            
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=5e-6,
    #         steps=1000,
    #         experiment_name=f"gtelarge_l120_s4_maxlen256_bs2_{model_name}",
    #         dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231004065532_gte-large_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #         freeze_layers=0,
    #         max_length=256,
    #         model_name=model_name,
    #         num_context=2,
    #         data_source_without=[
    #             "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #             "20230924083304_gpt5",
    #         ]
    #     )
    #     main(config=config)

    # for lr in [1e-6, 2e-6, 5e-6]:
    #     config = BertConfig(
    #         debug=False,
    #         epochs=3,
    #         batch_size=2,
    #         lr=lr,
    #         steps=1000,
    #         warmup_ratio=0.01,
    #         max_grad_norm=10,
    #         experiment_name=f"gtesmall_l90s3_maxlen256_bs2_lr{lr}_3epochs",
    #         dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231003051044_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #         freeze_layers=0,
    #         max_length=256,
    #         num_context=3,
    #         data_source_without=[
    #             "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #             "20230924083304_gpt5",
    #         ]
    #     )
    #     main(config=config)
    
    # for n_context, max_length in zip([4, 6], [256, 384]):
    #     config = BertConfig(
    #         debug=False,
    #         epochs=2,
    #         batch_size=2,
    #         lr=5e-6,
    #         steps=1000,
    #         experiment_name=f"gtesmall_l60s2_ncontext{n_context}_ml{max_length}_bs2",
    #         dataset_dir=f"index/l60_s2_new",  # ここに作ったデータ入れる
    #         freeze_layers=0,
    #         max_length=max_length,
    #         num_context=n_context,
    #         data_source_without=[
    #             "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #             "20230924083304_gpt5",
    #         ]
    #     )
    #     main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=1e-6,
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     max_grad_norm=10,
    #     experiment_name=f"gtebase_l120s4_maxlen256_ncontexts2_lr1e-6_4epochs",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231006125029_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length120_stride_sentence4_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=2,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "20230924083304_gpt5",
    #     ]
    # )
    # main(config=config)  
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=1e-6,
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     max_grad_norm=10,
    #     experiment_name=f"gtesmall_l90s3_maxlen256_ncontexts3_lr1e-6_4epochs_without_10",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231005133048_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=3,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "20230924083304_gpt5", "10",
    #     ]
    # )
    # main(config=config)  

    # config = BertConfig(
    #     debug=False,
    #     epochs=6,
    #     batch_size=2,
    #     lr=0.5e-6,
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     max_grad_norm=10,
    #     experiment_name=f"gtesmall_l90s3_maxlen256_ncontexts3_lr0.5e-6_4epochs",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231005133048_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=3,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "20230924083304_gpt5",
    #     ]
    # )
    # main(config=config)  
    
    # for model_name in [
    #     "OpenAssistant/reward-model-deberta-v3-large-v2",
    #     "microsoft/deberta-v3-large",
    # ]:            
    #     config = BertConfig(
    #         debug=False,
    #         epochs=4,
    #         batch_size=2,
    #         lr=1e-6,
    #         steps=1000,
    #         warmup_ratio=0.01,
    #         max_grad_norm=10,
    #         experiment_name=f"gtelarge_l90_s3_maxlen256_bs2_{model_name}",
    #         dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231007080623_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",
    #         freeze_layers=0,
    #         max_length=256,
    #         model_name=model_name,
    #         num_context=3,
    #         data_source_without=[
    #             "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #             "20230924083304_gpt5",
    #         ]
    #     )
    #     main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=2,
    #     batch_size=2,
    #     lr=5e-6,
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     max_grad_norm=10,
    #     experiment_name=f"gtelarge_l90_s3_maxlen256_bs2_without3",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231005133048_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=3,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "20230924083304_gpt5", "3",
    #     ]
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=1e-6,
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     max_grad_norm=10,
    #     experiment_name=f"gtesmall_l90_s3_maxlen256_bs2_without3789_msdeberta",
    #     model_name="microsoft/deberta-v3-large",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231005133048_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=3,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "20230924083304_gpt5", "3", "7", "8", "9"
    #     ]
    # )
    # main(config=config)
    

    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=1e-6,
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     max_grad_norm=10,
    #     experiment_name=f"gtesmall_l90_s3_maxlen192_bs2_without3789_msdeberta",
    #     model_name="microsoft/deberta-v3-large",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231005133048_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=192,
    #     num_context=2,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "20230924083304_gpt5", "3", "7", "8", "9"
    #     ]
    # )
    # main(config=config)
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=1e-6,
    #     model_name="OpenAssistant/reward-model-deberta-v3-large",
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     max_grad_norm=10,
    #     experiment_name=f"gtesmall_l90s3_maxlen256_ncontexts3_lr1e-6_4epochs_without_10",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231005133048_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=3,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "20230924083304_gpt5", "10",
    #     ]
    # )
    # main(config=config)  

    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=1e-6,
    #     model_name="deepset/deberta-v3-large-squad2",
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     max_grad_norm=10,
    #     experiment_name=f"gtesmall_l90s3_maxlen256_ncontexts3_lr1e-6_4epochs_squad2",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231005133048_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=3,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "20230924083304_gpt5",
    #     ]
    # )
    # main(config=config)


    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=1e-6,
    #     model_name="deepset/roberta-large-squad2",
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     max_grad_norm=10,
    #     experiment_name=f"gtesmall_l90s3_maxlen256_ncontexts3_lr1e-6_4epochs_squad2_roberta",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231005133048_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=3,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "20230924083304_gpt5",
    #     ]
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=1e-6,
    #     model_name="VMware/deberta-v3-large-mrqa",
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     max_grad_norm=10,
    #     experiment_name=f"gtesmall_l90s3_maxlen256_ncontexts3_lr1e-6_4epochs_squad2_roberta",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231005133048_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=3,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "20230924083304_gpt5",
    #     ]
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=1e-6,
    #     model_name="deepset/roberta-large-squad2",
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     max_grad_norm=10,
    #     experiment_name=f"gtesmall_l90s3_maxlen256_ncontexts3_lr1e-6_4epochs_squad2_roberta",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231005133048_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=3,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "20230924083304_gpt5", "10",
    #     ]
    # )
    # main(config=config)

    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=1e-6,
    #     model_name="deepset/deberta-v3-large-squad2",
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     max_grad_norm=10,
    #     experiment_name=f"gtesmall_l90s3_maxlen256_ncontexts3_lr1e-6_4epochs_squad2_all_ds",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231005133048_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=3,
    #     data_source_without=[
    #         "20230924083304_gpt5",
    #     ]
    # )
    # main(config=config)
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=1e-6,
    #     model_name="deepset/deberta-v3-large-squad2",
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     max_grad_norm=10,
    #     experiment_name=f"gtesmall_l90s3_maxlen256_ncontexts3_lr1e-6_4epochs_squad2_without3789",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231005133048_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=3,
    #     data_source_without=[
    #         "20230924083304_gpt5", "3", "7", "8", "9", 
    #          "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #     ]
    # )
    # main(config=config)
    
    
    # config = BertConfig(
    #     debug=False,
    #     epochs=4,
    #     batch_size=2,
    #     lr=1e-6,
    #     model_name="OpenAssistant/reward-model-deberta-v3-large",
    #     steps=1000,
    #     warmup_ratio=0.01,
    #     max_grad_norm=10,
    #     experiment_name=f"gtesmall_l90s3_maxlen256_ncontexts3_lr1e-6_4epochs_without_10",
    #     dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231005133048_gte-small_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",  # ここに作ったデータ入れる
    #     freeze_layers=0,
    #     max_length=256,
    #     num_context=3,
    #     data_source_without=[
    #         "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
    #         "20230924083304_gpt5", "10",
    #     ]
    # )
    # main(config=config)  

    config = BertConfig(
        debug=False,
        epochs=4,
        batch_size=2,
        lr=1e-6,
        steps=1000,
        warmup_ratio=0.01,
        max_grad_norm=10,
        experiment_name=f"gtelarge_l90_s3_maxlen256_bs2_wo10_bge",
        dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231008020744_bge-base-en-v1.5_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence2_drop_categoryTrue_all",
        freeze_layers=0,
        max_length=256,
        num_context=3,
        data_source_without=[
            "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
            "20230924083304_gpt5", "10",
        ]
    )
    main(config=config)

    config = BertConfig(
        debug=False,
        epochs=4,
        batch_size=2,
        lr=1e-6,
        steps=1000,
        warmup_ratio=0.01,
        max_grad_norm=10,
        experiment_name=f"gtelarge_l90_s3_maxlen256_bs2_wo3,10",
        dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231007080623_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",
        freeze_layers=0,
        max_length=256,
        num_context=3,
        data_source_without=[
            "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
            "20230924083304_gpt5", "3", "10",
        ]
    )
    main(config=config)


    config = BertConfig(
        debug=False,
        epochs=4,
        batch_size=2,
        lr=1e-6,
        steps=1000,
        warmup_ratio=0.01,
        max_grad_norm=10,
        experiment_name=f"gtelarge_l90_s3_maxlen256_bs2_wo3,7,9,10",
        dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231007080623_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",
        freeze_layers=0,
        max_length=256,
        num_context=3,
        data_source_without=[
            "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
            "20230924083304_gpt5", "3", "7", "9", "10",
        ]
    )
    main(config=config)
    
    
    config = BertConfig(
        debug=False,
        epochs=3,
        batch_size=2,
        lr=1e-6,
        steps=1000,
        warmup_ratio=0.01,
        max_grad_norm=10,
        experiment_name=f"gtelarge_l90_s3_maxlen384_bs2_wo3,7,9,10",
        dataset_dir=f"output/context_pipeline/stage1/exp009.py/20231007080623_gte-base_wikiall_without_sep_targetprompt_and_choice_without_sep_token_length90_stride_sentence3_drop_categoryTrue_all",
        freeze_layers=0,
        max_length=384,
        num_context=4,
        data_source_without=[
            "additional_data/MMLU/test.csv", "additional_data/MMLU/train.csv",
            "20230924083304_gpt5", "3", "7", "9", "10",
        ]
    )
    main(config=config)

