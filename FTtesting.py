from datasets import load_dataset, DatasetDict, Dataset
from transformers import(
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from transformers import(
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

import evaluate
import torch
import numpy as np

filename = "training.txt"
with open (filename, "a") as file:

    #load_dataset
    dataset = load_dataset('shawhin/imdb-truncated')
    dataset

    #display % of training data with label =1
    np.array(dataset['train']['label']).sum()/len(dataset['train']['label'])
    print(np.array(dataset['train']['label']).sum())

    model_checkpoint = 'distilbert-base-uncased'
    #model_checkpoint = 'roberta-base'

    #define label maps
    id2label = {0:"Negative", 1:"Positive"}
    label2id = {"Negative":0,"Positive":1}

    #generate classification model from model_checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)

    #create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,add_prefix_space=True)

    #create tokenize function
    def tokenize_function(examples):
        #extract text
        text = examples["text"] # "text" from the dataset object 

        #tokenize and truncate text
        tokenizer.truncation_side = "left"
        tokenized_inputs = tokenizer(
            text,
            return_tensors ="np",
            truncation = True,
            max_length = 512
        )

        return tokenized_inputs

    #add pad token if none exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token':'[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    #tokenize training and validation datasets
    tokenized_dataset = dataset.map(tokenize_function, batched = True)
    tokenized_dataset

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #import accuracy evaluation metric
    accuracy = evaluate.load("accuracy")

    #p = model output
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis = 1)

        return {"accuracy": accuracy.compute(predictions=predictions , references=labels)}

    #define list of examples
    text_list = ["It was good.", "Not a fan, do not recommend.","Better than the first one.","This is not worth watching even once.", "This one is a pass."]

    label1 = "Untrained model predictions:"
    label2 = "----------------------------"
    print(label1)
    print(label2)

    file.write(label1 + "\n")
    file.write(label2 + "\n")
        

    for text in text_list:
        #tokenize text
        inputs = tokenizer.encode(text, return_tensors='pt')
        #compute logits
        logits = model(inputs).logits
        #convert logits to label
        predictions = torch.argmax(logits)

        output1 = text + '-' + id2label[predictions.tolist()]
        print(output1)
        
        file.write(output1 + "\n")

    peft_config = LoraConfig(task_type = "SEQ_CLS", #sequence classification
                            r=4,#intrinsic rank of trainable weight matrix
                            lora_alpha=32, # learning rate
                            lora_dropout=0.01, # probability of dropout
                            target_modules = ['q_lin']) # we apply lora to query layer

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    lr = 1e-3
    batch_size = 4
    num_epochs=10

    #define training arguments
    training_args = TrainingArguments(
        output_dir= model_checkpoint + '-lora-text-classification',
        learning_rate = lr,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        num_train_epochs= num_epochs,
        weight_decay = 0.01,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end= True
    )

    trainer = Trainer(
        model = model, #peft model
        args = training_args, #hyper parameters
        train_dataset=tokenized_dataset['train'], # training data
        eval_dataset = tokenized_dataset["validation"], # validation data
        tokenizer = tokenizer, #define tokenizer
        data_collator = data_collator, # dynamically pad examples in each batch
        compute_metrics = compute_metrics, # evalute model using function defined earlier
    )

    trainer.train()

    model.to('cpu')

    print("Trained model predictions:")
    print("--------------------------")
    text_list = ["It was good.", "Not a fan, do not recommend.","Better than the first one.","This is not worth watching even once.", "This one is a pass.","Why did i waste my morning nap for this", "Sad that the ending was happy"]


    for text in text_list:
        inputs = tokenizer.encode(text,return_tensors = 'pt').to("cpu")

        logits = model(inputs).logits
        predictions = torch.max(logits,1).indices

        print(text + " - " + id2label[predictions.tolist()[0]])

    training_args.output_dir = './'  # Set desired output directory

    file.write("Model done training")