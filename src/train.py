import os
import json
import wandb
import time

import torch
from utils import create_logger, save_best, reset_wandb_env


def create_log_message(metrics_log, t):
  '''
  Various scenarios:
   - metrics_log['ari'] == None --> Epoch without cluster evaluation, report only train and validation loss
   - metrics_log['num_failures] != None --> Epoch with cluster validation of DAC model
   - metrics_log['ari'] != None but metrics_log['acc'] == None --> Epoch with cluster validation but for model without acc, tpr and tnr computation
     (PermEqui, ..)
   - else cluster validation for MIL or ABC, all metrics are to be reported.
  '''
  
  if metrics_log['val_loss'] is None:
    log_message = f'epoch {(t+1):5d}, train_loss: {metrics_log["train_loss"]:.4f}, ' \
                    f'early_stop_loss: {metrics_log["early_stop_loss"]:.4f}'
    
  elif metrics_log['val_ari'] is None:
    log_message = f'epoch {(t+1):5d}, train_loss: {metrics_log["train_loss"]:.4f}, ' \
                    f'val_loss: {metrics_log["val_loss"]:.4f}'
  
  elif metrics_log['val_num_failures'] is not None:
    log_message =  f'epoch {(t+1):5d}, train_loss: {metrics_log["train_loss"]:.4f}, val_loss: '\
                    f'{metrics_log["val_loss"]:.4f}, val_ari: {metrics_log["val_ari"]:.4f}, '\
                    f'val_nmi: {metrics_log["val_nmi"]:.4f}, val_num_failures: {metrics_log["val_num_failures"]}'

  elif metrics_log['val_acc'] is None:
    log_message =  f'epoch {(t+1):5d}, train_loss: {metrics_log["train_loss"]:.4f}, val_loss: '\
                    f'{metrics_log["val_loss"]:.4f}, val_ari: {metrics_log["val_ari"]:.4f}, '\
                    f'val_nmi: {metrics_log["val_nmi"]:.4f}'
  
  else:
    log_message =  f'epoch {(t+1):5d}, train_loss: {metrics_log["train_loss"]:.4f}, val_loss: '\
                    f'{metrics_log["val_loss"]:.4f}, val_ari: {metrics_log["val_ari"]:.4f}, '\
                    f'val_nmi: {metrics_log["val_nmi"]:.4f}, val_acc: {metrics_log["val_acc"]:.4f}, ' \
                    f'val_tpr: {metrics_log["val_tpr"]:.4f}, val_tnr: {metrics_log["val_tnr"]:.4f}'
    
  return log_message


def train_fn(args, run_epoch, eval_fn, create_model, create_dataloaders):
  
  reset_wandb_env()
  
  # W&B
  if args.n_folds:
    job_type = os.path.basename(os.path.normpath(args.save_dir))
    run_name = job_type+f'-{args.fold_number}'
    args.save_dir = os.path.join(args.save_dir, f'{args.fold_number}')
    
    # run = wandb.init(name=run_name, config=args, project='pdf-clustering', tags=[args.model_type], group=args.experiment, job_type=job_type, settings=wandb.Settings(start_method="fork"))
    run = wandb.init(name=run_name, config=args, project='pdf-clustering', tags=[args.model_type], group=args.experiment, job_type=job_type) 
    
  else:
    
    run = wandb.init(name=args.save_dir[8:], config=args, group=args.experiment, project='pdf-clustering', tags=[args.model_type])
    
  # Create save_dir
  if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

  # Save args
  with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, sort_keys=True, indent=2)

  # Logging
  logger = create_logger(args.save_dir)
  if args.n_folds:
    logger.info(f'fold: {args.fold_number+1}/{args.n_folds}')
  
  # gpu
  if torch.cuda.is_available():
    device = 'cuda'
  else:
    device = 'cpu'

  # Data
  train_dl, early_stop_dl, val_dl, val_dl_cluster, d_len = create_dataloaders(args)
  logger.info(f'train dataset: {d_len[0]} sets')
  logger.info(f'dataset for early stopping: {d_len[1]} sets')
  logger.info(f'validation dataset: {d_len[2]} sets')
  
  # Prepare early stop
  stopped = False
  best_epoch = 0
  best_metrics_log = {}
  best_loss = torch.Tensor([float('Inf')])
  
  model, criterion, optimizer = create_model(args)
  model = model.to(device)
  
  run.watch(model)

  # Print args
  logger.info('using args: \n' + json.dumps(vars(args), sort_keys=True, indent=2))

  # Measure time to train model (count until best metrics achieved)
  tick = time.time()
  # epochs
  for t in range(args.n_epochs):

    # Run epoch
    metrics_log = run_epoch(t, args, model, criterion, optimizer, train_dl, early_stop_dl, val_dl, val_dl_cluster, eval_fn, device)
        
    # Print log
    if ((t+1)%args.print_freq == 0) or (t==0):
      log_message = create_log_message(metrics_log, t)
      logger.info(log_message)

    # W&B
    run.log(metrics_log, step=t)
      
    # Save best model
    if metrics_log['early_stop_loss'] < best_loss:
      best_loss, best_epoch = metrics_log['early_stop_loss'], t
      
      # Best
      best_metrics_log = metrics_log
      best_metrics_log['time_to_best_metrics'] = time.time() - tick
      save_best(args, t, model, best_metrics_log)

    # Check early stop
    if t >= best_epoch + args.early_stop:
      logger.info('EARLY STOP')
      break
      
  # End of training -> compute and log best validation metrics
  logger.info(f"Training ended: loading best model and computing it's metrics.")

  checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pt.tar'))
  model.load_state_dict(checkpoint['model'])

  time_to_best_metrics = best_metrics_log['time_to_best_metrics']
  tick_eval = time.time()
  best_metrics_log = eval_fn(model, criterion, val_dl_cluster, args, device, True)
  time_to_eval = time.time()-tick_eval
                      
  # Log best validation metrics
  best_metrics_log_edited = {}
  for entry in best_metrics_log:
    best_metrics_log_edited['best_'+entry] = best_metrics_log[entry]

  best_metrics_log['time_to_best_metrics'] = time_to_best_metrics
  best_metrics_log['time_to_eval'] = time_to_eval
  best_metrics_log['time'] = time.time() - tick
  
  logger.info(f'Metrics for best early stop loss:\n {best_metrics_log}')
  wandb.log(best_metrics_log_edited, step=t+1)
  wandb.log(best_metrics_log, step=t+1)
  
  # Only save model if args.save_model is set to True (to save memory space on RCI)
  if args.save_model:
    logger.info('Training finished successfully. Best model is saved at {}'.format(
        os.path.join(args.save_dir, 'checkpoint.pt.tar')))
    
    # Save metrics as well
    checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pt.tar'))
    checkpoint['best_metrics_log'] = best_metrics_log
    
    torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pt.tar'))
    logger.info('Final metrics save done.')
    
  else:
    logger.info('Training finished successfully.')
    
    # Delete model to save space
    checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pt.tar'))
    checkpoint['model'] = None
    
    # Save only metrics
    checkpoint['best_metrics_log'] = best_metrics_log
    torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pt.tar'))
    logger.info('Final metrics save done.')

  run.join()
  logger.handlers = []
