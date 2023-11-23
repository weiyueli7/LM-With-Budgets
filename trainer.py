import json
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaModel, LlamaConfig, TrainingArguments, DataCollatorForLanguageModeling, Trainer


from data import *



def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()



def train():

    with open('configs/config.json', 'r') as file:
        config_dict = json.load(file)

    # Create the configuration object
    config = LlamaConfig.from_dict(config_dict)

    # Initialize the LlamaForCausalLM model
    model = LlamaForCausalLM(config)


    dataset_path = '/home/shawn/nvme/vl_research/jerry-agent/SlimPajama-6B/data'
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

    # File paths for lazy loading
    train_file_paths = get_file_paths(dataset_path, 'train')
    valid_file_paths = get_file_paths(dataset_path, 'valid')
    test_file_paths = get_file_paths(dataset_path, 'test')


    train_dataset = SlimPajamaDataset(train_file_paths, tokenizer)
    valid_dataset = SlimPajamaDataset(valid_file_paths, tokenizer)
    test_dataset = SlimPajamaDataset(test_file_paths, tokenizer)


    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=50,  
        report_to="all"  

    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator
    )

    # Train
    trainer.train()


    # Evaluate
    trainer.evaluate()


    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()