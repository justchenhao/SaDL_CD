import sys
import os
import time
import numpy as np


class NpyLogger():
    def __init__(self,
                 save_dir: str,
                 log_name: str = 'train_val_acc.npy'):
        self.save_dir = save_dir
        self.outfile = os.path.join(self.save_dir, log_name)
        self.items = {}

    def log(self,
            item: dict,
            step: int = None):
        """
        记录键值对，比如：
            epoch_id: 1
            step_id: 1
            train_acc: XX
            val_acc: XX
        :param item:
        :return:
        """
        for k, v in item.items():
            self.items[k] = np.append(self.items.get(k, []), v)
        self.save()

    def save(self):
        np.save(self.outfile, self.items)


class WandbLogger():
    def __init__(self,
                 save_dir: str,
                 project: str = 'test',
                 name: str = 'name',
                 resume: str = 'auto'
                 ):
        import wandb
        wandb.init(dir=save_dir,
                   project=project,
                   name=name,
                   resume=resume)

    def define_metric(self,
                      metric_name: str = 'acc',
                      summary: str = 'max'):
        """
        定义评价指标，可以用于筛选最好模型
        :param metric_name:
        :param summary:
        :return:
        """
        import wandb
        wandb.define_metric(metric_name, summary=summary)

    def log(self,
            item: dict,
            step: int = None):
        import wandb
        wandb.log(item, step=step)

    def __del__(self):
        print('脚本运行结束, 释放windb内存')
        import wandb
        wandb.finish()


class Logger(object):
    def __init__(self,
                 outfile,
                 with_npy: bool = False,
                 with_wandb: bool = False,
                 **kwargs):
        self.terminal = sys.stdout
        self.log_path = outfile
        now = time.strftime("%c")
        self.write('================ (%s) ================\n' % now)
        self.with_npy = with_npy
        self.with_wandb = with_wandb
        self.metric_loggers = []
        save_dir = os.path.dirname(outfile)
        if with_npy:
            npy_name = kwargs.get('npy_name', 'train_val_acc.npy')
            self.metric_loggers.append(NpyLogger(save_dir, npy_name))
        if with_wandb:
            project = kwargs.get('project_task', 'test')
            name = kwargs.get('project_name', 'test')
            name = kwargs.get('wandb_name', name)
            self.metric_loggers.append(WandbLogger(save_dir,
                                                   project=project,
                                                   name=name))

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_path, mode='a') as f:
            f.write(message)

    def write_dict(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %.7f ' % (k, v)
        self.write(message)

    def write_dict_str(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %s ' % (k, v)
        self.write(message)

    def flush(self):
        self.terminal.flush()

    def log(self, items: dict, step: int = None):
        for metric_logger in self.metric_loggers:
            metric_logger.log(items, step)

    def define_metric(self,
                      metric_name: str = 'acc',
                      summary: str = 'max'):
        for metric_logger in self.metric_loggers:
            metric_logger.define_metric(metric_name, summary)


class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)

    def str_estimated_complete(self):
        return str(time.ctime(self.est_finish))

    def str_estimated_remaining(self):
        return str(self.est_remaining/3600) + 'h'

    def estimated_remaining(self):
        return self.est_remaining/3600

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()

    def lapse(self):
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out

