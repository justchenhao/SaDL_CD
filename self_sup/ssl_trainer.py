import numpy as np
import os
from typing import Dict
import torch

from self_sup import define_G
from models.schedulers import get_scheduler
from models.optimizers import get_optimizer

from misc.metric_tool import AverageMeter
from misc.logger_tool import Logger, Timer


# Decide which device we want to run on
# torch.cuda.current_device()


class BaseTrainer():

    def __init__(self, args, dataloaders):
        self.dataloaders = dataloaders

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)
        self.args = args
        # define some other vars to record the training states

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_loss = 10000
        self.best_val_loss = 10000.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.num_epoch_to_save_best = args.__dict__.get('num_epoch_to_save_best', None)

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)

    def _load_checkpoint(self):
        if os.path.exists(os.path.join(self.checkpoint_dir, 'last_ckpt.pt')):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'last_ckpt.pt'),
                                    map_location='cpu')

            # update model states
            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(
                checkpoint['lr_scheduler_state_dict'])

            self.model.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_loss = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_loss, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')

    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        # print(est)
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        # print(imps)
        return imps, est

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_loss': self.best_val_loss,
            'best_epoch_id': self.best_epoch_id,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')

        self.logger.write('Lastest model updated. Epoch_loss=%.4f, Historical_best_loss=%.4f (at epoch %d)\n'
              % (self.epoch_loss, self.best_val_loss, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval loss)
        if self.epoch_loss < self.best_val_loss:
            self.best_val_loss = self.epoch_loss
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

        if self.num_epoch_to_save_best:
            if self.epoch_id % self.num_epoch_to_save_best == self.num_epoch_to_save_best - 1:
                import shutil
                best_path = os.path.join(self.checkpoint_dir, 'best_ckpt.pt')
                out_path = os.path.join(self.checkpoint_dir,
                                        f'best_ckpt_epoch_{self.best_epoch_id}.pt')
                shutil.copy(best_path, out_path)


class SSLTrainer(BaseTrainer):
    def __init__(self, args, dataloaders):
        super().__init__(args, dataloaders)

        self.model = define_G(args)

        # optimizer
        self.lr = args.lr
        self.optim_mode = args.optim_mode
        self.lr_policy = args.lr_policy
        self.configure_optimizers()

        self.running_metric = AverageMeter()

        self.loss_dict = {}
        self.VAL_LOSS = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_loss.npy')):
            self.VAL_LOSS = np.load(os.path.join(self.checkpoint_dir, 'val_loss.npy'),allow_pickle=True)
        self.TRAIN_LOSS = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_loss.npy')):
            self.TRAIN_LOSS = np.load(os.path.join(self.checkpoint_dir, 'train_loss.npy'),allow_pickle=True)

    def _update_metric(self):
        """
        update metric
        """
        self.running_metric.update(self.loss_dict['loss'].item())

    def _clear_cache(self):
        self.running_metric.clear()

    def _update_lr_schedulers(self, state='epoch'):
        if (self.lr_policy == 'poly' and state == 'step') or\
                (self.lr_policy != 'poly' and state == 'epoch'):
            self.lr_scheduler.step()

    def _collect_running_batch_states(self):

        running_loss = self._update_metric()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, ' %\
                      (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     imps*self.batch_size, est,)
            loss_message = ''
            for k, v in self.loss_dict.items():
                loss_message += '%s: %.5f, ' % (k, v.item())
            message = message + loss_message + '\n'
            self.logger.write(message)

    def _collect_epoch_states(self):
        self.epoch_loss = self.running_metric.avg
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_loss= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_loss))
        self.logger.write('\n')

    def _on_before_zero_grad(self):
        self.model.on_before_zero_grad()

    def _backward_G(self):
        self.loss_dict['loss'].backward()

    def transfer_batch_to_device(self, batch: Dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = batch[k].to(self.device)

    def train_models(self):
        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.model.train()  # Set model to training mode
            # Iterate over data.
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                self.transfer_batch_to_device(batch)
                batch = self.model.on_after_batch_transfer(batch)
                self.loss_dict = self.model.training_step(batch, self.batch_id)
                # self._forward_pass(batch)
                # update G
                self._on_before_zero_grad()
                self.optimizer.zero_grad()
                self._backward_G()
                self.optimizer.step()
                self._update_lr_schedulers(state='step')

                self._collect_running_batch_states()
                self._timer_update()

            self._collect_epoch_states()
            self._update_training_loss_curve()
            self._update_lr_schedulers(state='epoch')

            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.model.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self.transfer_batch_to_device(batch)
                    batch = self.model.on_after_batch_transfer(batch)
                    self.loss_dict = self.model.training_step(batch, self.batch_id)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_loss_curve()
            self._update_checkpoints()

    def _update_val_loss_curve(self):
        # update val loss curve
        self.VAL_LOSS = np.append(self.VAL_LOSS, [self.epoch_loss])
        np.save(os.path.join(self.checkpoint_dir, 'val_loss.npy'), self.VAL_LOSS)

    def _update_training_loss_curve(self):
        # update train loss curve
        self.TRAIN_LOSS = np.append(self.TRAIN_LOSS, [self.epoch_loss])
        np.save(os.path.join(self.checkpoint_dir, 'train_loss.npy'), self.TRAIN_LOSS)

    def configure_optimizers(self):
        from models.optimizers import get_params_groups
        lr_multi = 1
        if hasattr(self.args, 'lr_multi'):
            lr_multi = self.args.lr_multi
        params_list = get_params_groups(self.model, lr=self.lr, lr_multi=lr_multi)
        self.optimizer = get_optimizer(params_list, optim_mode=self.optim_mode,
                                          lr=self.lr, lr_policy=self.lr_policy,
                                         init_step=self.global_step, max_step=self.total_steps)
        print('get optimizer %s' % self.optim_mode)
        self.lr_scheduler = get_scheduler(self.optimizer, self.lr_policy,
                                                max_epochs=self.max_num_epochs,
                                                steps_per_epoch=self.steps_per_epoch)
