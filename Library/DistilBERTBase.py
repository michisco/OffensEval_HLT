import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertForSequenceClassification, DistilBertConfig, DistilBertModel, DistilBertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from AnalysisGraph import show_confusion_matrix, show_report, plot_curves, show_reportNoTorch, show_confusion_matrixNoTorch

import numpy as np
import random

import time

#Random seed initialization for replicability of results
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, model_name = "bert-base-uncased",
                    unit_layers = [768, 64, 32, 2],
                    dropout = 0.7):
        """
        @param    bert: a BertModel object
        @param    model_name: hugginface transformers name
        @param    unit_layers: units network
        @param    dropout: dropout's number
        """
        super(BertClassifier, self).__init__()

        # Instantiate BERT model
        self.bert = DistilBertModel.from_pretrained(model_name)

        modules = []
        modules.append(nn.Linear(unit_layers[0], unit_layers[1]))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout))
        for i in range(1, len(unit_layers) - 2):
            modules.append(nn.Linear(unit_layers[i], unit_layers[i+1]))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        
        modules.append(nn.Linear(unit_layers[len(unit_layers) - 2], unit_layers[len(unit_layers) - 1]))
        self.classifier = nn.Sequential(*modules)

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

def initialize_model(hparams, len_data, num_warmup_steps = 0):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(hparams['model_name'], hparams['unit_layers'], hparams['dropout'])
    print(bert_classifier.classifier)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(hparams['device'])

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=hparams['learning_rate'],  
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len_data * hparams['num_epochs']

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

def f1_scoreEval(preds, labels):
    preds_copy = torch.tensor(preds)
    preds_flat = np.argmax(preds_copy.cpu(), axis=1).flatten()
    labels_flat = labels.cpu().flatten()
    return f1_score(labels_flat, preds_flat, average='macro')
    
def trainBERT(model, train_dataloader, optimizer, scheduler, hparams, val_dataloader=None, evaluation=False):
    """Train the BertClassifier model."""
    loss_ce = hparams['loss']
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs_list = []
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(hparams['num_epochs']):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'F1 Train':^9} | {'Val Loss':^10} | {'Val Acc':^9} | {'F1 Val':^9} | {'Elapsed':^9}")
        print("-"*95)

        epochs_list.append(epoch_i)
        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts, f1_value_train_batch, f1_value_train_tot  = 0, 0, 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(hparams['device']) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            loss = loss_ce(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()
            f1_value_train_batch+= f1_scoreEval(logits, b_labels) 
            f1_value_train_tot+= f1_scoreEval(logits, b_labels) 

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {f1_value_train_batch / batch_counts:^9.2f} | {'-':^10} | {'-':^9} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts, f1_value_train_batch = 0, 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        avg_f1_value = f1_value_train_tot / len(train_dataloader)
        
        train_losses.append(avg_train_loss)
        train_accs.append(avg_f1_value)
        print("-"*95)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on validation set.
            val_loss, val_accuracy, f1_value_validation = evaluate(model, val_dataloader, hparams['device'], loss_ce)

            val_losses.append(val_loss)
            val_accs.append(val_accuracy)
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {avg_f1_value:^9.2f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {f1_value_validation:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*95)
        print("\n")
    
    print("Training complete!")
    return train_losses, val_losses, train_accs, val_accs, epochs_list

def evaluate(model, val_dataloader, device, loss_ce):
    """After the completion of each training epoch, measure the model's performance
    on validation set."""
    
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    b_input_ids = torch.tensor([], dtype=torch.long)
    b_attn_mask = torch.tensor([], dtype=torch.long)
    b_labels = torch.tensor([],dtype=torch.long)

    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids_curr, b_attn_mask_curr, b_labels_curr = tuple(t for t in batch)

        # Concat input and labels
        b_input_ids = torch.cat((b_input_ids, b_input_ids_curr), 0)
        b_attn_mask = torch.cat((b_attn_mask, b_attn_mask_curr), 0)
        b_labels = torch.cat((b_labels, b_labels_curr), 0)

    b_input_ids = b_input_ids.to(device)
    b_attn_mask = b_attn_mask.to(device)

    # Compute logits
    with torch.no_grad():
        logits = model(b_input_ids, b_attn_mask)

    del b_input_ids
    del b_attn_mask
    torch.cuda.empty_cache()
    
    # Compute loss
    b_labels = b_labels.to(device)
    loss = loss_ce(logits, b_labels)
    val_loss = loss.item()
    # Get the predictions
    preds = torch.argmax(logits, dim=1).flatten()
    
    # Calculate the accuracy rate
    val_accuracy = (preds == b_labels).cpu().numpy().mean() * 100
    f1_value = f1_scoreEval(logits, b_labels) * 100

    # Compute the average accuracy and loss over the validation set.
    return val_loss, val_accuracy, f1_value

def evaluateFinal(model, val_dataloader, device, loss_ce, label_name = ["Not Off", "Off"]):
    """After the completion of each training epoch, measure the model's performance
    on validation set."""
    
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    b_input_ids = torch.tensor([], dtype=torch.long)
    b_attn_mask = torch.tensor([], dtype=torch.long)
    b_labels = torch.tensor([],dtype=torch.long)

    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids_curr, b_attn_mask_curr, b_labels_curr = tuple(t for t in batch)

        # Concat input and labels
        b_input_ids = torch.cat((b_input_ids, b_input_ids_curr), 0)
        b_attn_mask = torch.cat((b_attn_mask, b_attn_mask_curr), 0)
        b_labels = torch.cat((b_labels, b_labels_curr), 0)

    b_input_ids = b_input_ids.to(device)
    b_attn_mask = b_attn_mask.to(device)

    # Compute logits
    with torch.no_grad():
        logits = model(b_input_ids, b_attn_mask)

    del b_input_ids
    del b_attn_mask
    torch.cuda.empty_cache()
    
    # Compute loss
    b_labels = b_labels.to(device)
    loss = loss_ce(logits, b_labels)
    val_loss = loss.item()
    # Get the predictions
    preds = torch.argmax(logits, dim=1).flatten()

    # Display confusion matrix and report
    show_confusion_matrix(b_labels, preds, label_name)
    show_report(b_labels, preds, label_name)
    
    # Calculate the accuracy rate
    val_accuracy = (preds == b_labels).cpu().numpy().mean() * 100
    f1_value = f1_scoreEval(logits, b_labels) * 100

    # Compute the average accuracy and loss over the validation set.
    return val_loss, val_accuracy, f1_value

def evaluate_for_kfold(model, val_dataloader, device, loss_ce):
    """After the completion of each training epoch, measure the model's performance
    on validation set."""
    
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    preds = []
    # Tracking variables
    val_accuracy = []
    val_loss = []
    f1_value = []

    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        # Compute loss
        loss = loss_ce(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds+=(torch.argmax(logits, dim=1).flatten().tolist())
    
    return preds

def training_with_kfold(input_ids_train, attention_masks_train, dataset_kfold_labels, hparams):
  number_of_splits = hparams['folds']
  index_model = 0
  
  losses = {}
  losses['train'] = []
  losses['val']  = []
  accuracies = {}
  accuracies['train'] = []
  accuracies['val']  = []
  epochs_list = []
  
  cv_kfold = StratifiedKFold(n_splits=number_of_splits, shuffle=True, random_state=100)
  models = []
  epoch_full_model = hparams['num_epochs']
  for train_index, validation_index in cv_kfold.split(input_ids_train, dataset_kfold_labels):
    num_train_steps = int(len(input_ids_train) / hparams['batch_size'] * (epoch_full_model))+1
    num_warmup_steps = int((epoch_full_model) * hparams['warmup_proportion'])
    bert_classifier, optimizer, scheduler = initialize_model(hparams, num_warmup_steps= num_warmup_steps, len_data = len(input_ids_train))

    # Create the DataLoader for training set
    train_data_kfold = TensorDataset(input_ids_train[train_index], attention_masks_train[train_index], dataset_kfold_labels[train_index])
    train_sampler_kfold = RandomSampler(train_data_kfold)
    train_dataloader_kfold = DataLoader(train_data_kfold, sampler=train_sampler_kfold, batch_size=hparams['batch_size'])

    # Create the DataLoader for validation set
    val_data_kfold = TensorDataset(input_ids_train[validation_index], attention_masks_train[validation_index], dataset_kfold_labels[validation_index])
    val_sampler_kfold = SequentialSampler(val_data_kfold)
    val_dataloader_kfold = DataLoader(val_data_kfold, sampler=val_sampler_kfold, batch_size=hparams['batch_size'])

    losses['train'], losses['val'], accuracies['train'], accuracies['val'], epochs_list = trainBERT(bert_classifier, train_dataloader_kfold, optimizer, scheduler, hparams, val_dataloader_kfold, evaluation=True)

    models.append(bert_classifier)
    plot_curves(losses, accuracies, index_model, epochs_list)
    index_model = index_model + 1
  return models

def predict(models, test_dataloader, hparams):
  # Make predictions
  results = []
  y_predict = []
  for model in models:
    model.to(hparams['device'])
    y_predict.append(evaluate_for_kfold(model, test_dataloader, hparams['device'], hparams['loss']))
    model.to("cpu")

  y_predict = np.array(y_predict)

  #Argmax across classes
  for i in range(y_predict.shape[1]):
    counts = np.bincount(y_predict[:,i])
    results.append(np.argmax(counts))
    
  return results
 
def run_with_kfold(in_ids_train, att_mask_train, test_dataloader, db_kfold_labels, test_labels, hparams = None, label_name = ["Not Off", "Off"]):

    models = training_with_kfold(in_ids_train, att_mask_train, db_kfold_labels, hparams)
    y_test_pred = predict(models, test_dataloader, hparams)

    print("------RESULT------")
    # Display confusion matrix and report
    show_confusion_matrixNoTorch(test_labels, y_test_pred, label_name)
    show_reportNoTorch(test_labels, y_test_pred, label_name)
    
    print("f1_score test tweets: {}".format(f1_score(test_labels, y_test_pred,average="macro")))
    return models, y_test_pred
