import math
import torch
from torch.utils.tensorboard import SummaryWriter
import subprocess
import numpy as np
import os
import dill
import time
import copy
import json

class MetricSaverBase(object):
    def __init__(self, adding_attributes=None):
        self._mean_acc = None
        self._best_acc = None
        self._best_loss = None
        self._train_loss = []
        self._train_acc = []
        self._eva_loss = []
        self._eva_acc = []
        self._test_acc = []
        self._test_loss = []
        if adding_attributes is not None:
            for l in adding_attributes:
                setattr(self, l, None)

    @property
    def strict_best_acc(self):
        """采用严格模式的acc，也就是取验证集上loss最小的测试集结果

        Returns:
            [type] -- [description]
        """
        if len(self._test_acc) == 0:
            self._best_acc = 0
        else:
            min_index = np.argmin(self._eva_loss)
            self._best_acc = self._test_acc[min_index]
        return self._best_acc
    
    @property
    def best_acc(self):
        """采用非严格模式的acc，也就是只取验证集上的最大值

        Returns:
            [type] -- [description]
        """
        if len(self._eva_acc) == 0:
            self._best_acc = 0
        else:
            # min_index = np.argmin(self._eva_acc)
            self._best_acc = max(self._eva_acc)
        return self._best_acc
    
    @property
    def mean_acc(self):
        if len(self._eva_acc) == 0:
            self._mean_acc = 0
        else:
            self._mean_acc = np.mean(self._eva_acc)
        return self._mean_acc
    
    @property
    def train_acc(self):
        if len(self._train_acc) == 0:
            return 0
        else:
            return self._train_acc[-1]
    
    @property
    def train_loss(self):
        if len(self._train_loss) == 0:
            return 0
        else:
            return self._train_loss[-1]
    
    @property
    def eva_acc(self):
        if len(self._eva_acc) == 0:
            return 0
        else:
            return self._eva_acc[-1]
    
    @property
    def eva_loss(self):
        if len(self._eva_loss) == 0:
            return 0
        else:
            return self._eva_loss[-1]
    
    @property
    def test_acc(self):
        if len(self._test_acc) == 0:
            return 0
        else:
            return self._test_acc[-1]
    
    @property
    def test_loss(self):
        if len(self._test_loss) == 0:
            return 0
        else:
            return self._test_loss[-1]
    
    @property
    def min_loss(self):
        if len(self._test_loss) == 0:
            return 0
        else:
            return min(self._test_loss)
    
    def add_record(self, 
                    epoch_train_acc, 
                    epoch_train_loss, 
                    epoch_eva_acc=None, 
                    epoch_eva_loss=None, 
                    epoch_test_acc=None, 
                    epoch_test_loss=None, 
                    monitor='eva_loss', 
                    mode='min'):

        # 这里是为了判断新加入的acc是否是best_acc
        flag = False
        if monitor == 'eva_acc' and mode == 'max' and epoch_eva_acc > self.best_acc:
            flag = True
        if monitor == 'eva_loss' and mode == 'min' and epoch_eva_loss < self.min_loss:
            flag = True

        self._train_acc.append(epoch_train_acc)
        self._train_loss.append(epoch_train_loss)
        if epoch_eva_acc is not None:
            self._eva_acc.append(epoch_eva_acc)
        if epoch_eva_loss is not None:
            self._eva_loss.append(epoch_eva_loss)
        if epoch_test_acc is not None:
            self._test_acc.append(epoch_test_acc)
        if epoch_test_loss is not None:
            self._test_loss.append(epoch_test_loss)
        return flag
    
    def __call__(self, *args, **kwargs):
        return self.add_record(*args, **kwargs)
    
    def __repr__(self):
        return "train_acc:{:.4f}, train_loss:{:.4f}, eva_acc:{:.4f}, eva_loss:{:4f}, test_acc:{:.4f}, test_loss:{:.4f}| best_acc:{:.4f},\
 mean_acc:{:.4f}".format(self.train_acc, self.train_loss, self.eva_acc, self.eva_loss, self.test_acc, self.test_loss, self.best_acc, self.mean_acc)

    def close(self):
        return
    
    def start(self):
        return

    @property
    def has_test(self):
        return len(self._test_acc)

    def save(self, dir_name='.', prefix="", suffix=""):
        if len(self._test_acc):
            file_name = prefix + "{:.4f}-{:.4f}".format(self.strict_best_acc, self.best_acc) + suffix
        else:
            file_name = prefix + "{:.4f}".format(self.best_acc) + suffix
        dir_name = os.path.join(dir_name, file_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            # print(f"made new {dir_name} here.")
        with open(f'{dir_name}/train_acc.pkl', 'wb') as f:
            dill.dump(self._train_acc, f)
        with open(f'{dir_name}/train_loss.pkl', 'wb') as f:
            dill.dump(self._train_loss, f)
        with open(f'{dir_name}/eva_acc.pkl', 'wb') as f:
            dill.dump(self._eva_acc, f)
        with open(f'{dir_name}/eva_loss.pkl', 'wb') as f:
            dill.dump(self._eva_loss, f)
        # 如果存在测试集，则同时保存两个结果
        if len(self._test_acc):
            with open(f'{dir_name}/test_acc.pkl', 'wb') as f:
                dill.dump(self._test_acc, f)
            with open(f'{dir_name}/test_loss.pkl', 'wb') as f:
                dill.dump(self._test_loss, f)
        if len(self._test_acc):
            ans = {'strict_acc': self.strict_best_acc,
                    'best_acc': self.best_acc,
                    'mean_acc': self.mean_acc}
            with open('{}/{:.4f}-{:.4f}.json'.format(dir_name, self.strict_best_acc, self.best_acc), 'w') as f:
                json.dump(ans, f)
        else:
            ans = {'mean_acc': self.mean_acc,
                    'best_acc': self.best_acc}
            with open('{}/{:.4f}.json'.format(dir_name, self.best_acc), 'w') as f:
                json.dump(ans, f)
    
    def EarlyStopping(self, monitor='acc', warmup=50, min_delta=0.01, patience=20, strict=None):
        """[summary]

        Keyword Arguments:
            monitor {str} -- [用哪个值观测是否早停] (default: {'acc'})
            warmup {int} -- [从哪一轮开始观测早停] (default: {50})
            min_delta {float} -- [变动的容忍值] (default: {0.01})
            patience {int} -- [连续多少轮没有变动就早停] (default: {20})
            strict {[type]} -- [description] (default: {None})

        Returns:
            bool - True: stop, False: continue
        """
        assert monitor in ['acc', 'loss'], "your monitor is not right"
        assert isinstance(patience, int) and patience >= 2, "patience must be int and should bigger than two"
        
        if monitor == 'loss':
            return self.ES_loss(warmup, min_delta, patience, strict)
        elif monitor == 'acc':
            return self.ES_acc(warmup, min_delta, patience, strict)

    def ES_acc(self, warmup, min_delta, patience, strict):
        """[summary]

        Arguments:
            strict {[type]} -- 是否在有test集合的情况下使用test集合作为早停的标准

        Returns:
            [type] -- [description]
        """

        if len(self._test_acc) and strict in [None, True]: # 在有test集合下默认使用test集合的acc来判断
            check_ans = self._test_acc
        else:
            check_ans = self._eva_acc

        if len(check_ans) < warmup + patience: # 判断是否满足了要求的开始早停的轮数
            return False
        else:
            check_ans = check_ans[:patience]

        # 获取patience列表中前半和后半的数
        middle = patience // 2
        front_ans = check_ans[:middle]
        rear_ans = check_ans[middle:]
        # 早停的逻辑是：前半结果的极值和后半结果的极值的差值小于min_delta
        if (np.max(rear_ans) - np.min(front_ans)) <= min_delta:
                return True
        return False
    
    def ES_loss(self, warmup, min_delta, patience, strict):
        if len(self._test_loss) and strict in [None, True]: # 在有test集合下默认使用test集合的acc来判断
            check_ans = self._test_loss
        else:
            check_ans = self._eva_loss

        if len(check_ans) < warmup + patience: # 判断是否满足了要求的开始早停的轮数
            return False
        else:
            check_ans = check_ans[:patience]
        
        # 获取patience列表中前半和后半的数
        middle = patience // 2
        front_ans = check_ans[:middle]
        rear_ans = check_ans[middle:]
        # 早停的逻辑是：前半结果的极值和后半结果的极值的差值小于min_delta
        if np.max(front_ans) - np.min(rear_ans) <= min_delta:
            return True
        return False

    

class TensorBoardSaver(MetricSaverBase):
    def __init__(self, log_dir='runs/', file_name=None, adding_attributes=None, tag='fold_0'):
        super(TensorBoardSaver, self).__init__(adding_attributes)
        self.timestamp = time.strftime('%Y%m%d-%H_%M_%S', time.localtime(time.time()))
        if file_name == None:
            file_name = self.timestamp
        self.dir_name = os.path.join(log_dir, file_name)
        # self.writer = SummaryWriter(dir_name)
        self.tag = tag
    
    def start(self, start_server=False):
        self.writer = SummaryWriter(self.dir_name)
        if start_server:
            self.start_server()

    def start_server(self):
        self.subp = subprocess.Popen([f'tensorboard --logdir {self.dir_name} --bind_all'], shell=True)
        time.sleep(5)
    
    def close_server(self):
        try:
            self.subp.kill()
        except AttributeError:
            pass
    

    def add_record(self, epoch_train_acc, epoch_train_loss, epoch_eva_acc=None, epoch_eva_loss=None, epoch_test_acc=None, epoch_test_loss=None):
        flag = super().add_record(epoch_train_acc, epoch_train_loss, epoch_eva_acc, epoch_eva_loss, epoch_test_acc, epoch_test_loss)
        tag = self.tag

        self.writer.add_scalar(f"train_acc/train_acc_{tag}", epoch_train_acc, len(self._train_acc)-1)
        self.writer.add_scalar(f"train_loss/train_loss_{tag}", epoch_train_loss, len(self._train_loss)-1)
        if epoch_eva_acc is not None:
            self.writer.add_scalar(f"eva_acc/eva_acc_{tag}", epoch_eva_acc, len(self._eva_acc)-1)
        if epoch_eva_loss is not None:
            self.writer.add_scalar(f"eva_loss/eva_loss_{tag}", epoch_eva_loss, len(self._eva_loss)-1)
        if epoch_test_acc is not None:
            self.writer.add_scalar(f"test_acc/test_acc_{tag}", epoch_test_acc, len(self._test_acc)-1)
        if epoch_test_loss is not None:
            self.writer.add_scalar(f"test_loss/test_loss_{tag}", epoch_test_loss, len(self._test_loss)-1)
        return flag
    
    def close(self, close_tb=False):
        self.writer.close()
        if close_tb:
            self.close_server()
    
    def __del__(self):
        try:
            self.writer.close()
            self.subp.kill()
        except AttributeError:
            pass

class FoldMetricBase(object):
    """[summary]

    Keyword Arguments:
        k_fold {int} -- [description] (default: {10})
        saver {[type]} -- [description] (default: {MetricSaverBase})
        dir_name {str} -- [description] (default: {'.'})
        file_name {str} -- [description] (default: {'fold_test'})
        timestamp {bool} -- [description] (default: {True})
        suffix {[type]} -- [description] (default: {None})
        tb_server {bool} -- [description] (default: {False})
        adding_attributes {[type]} -- [description] (default: {None})
    """         
    def __init__(self,  
                k_fold=10, 
                saver=MetricSaverBase, 
                dir_name='.', 
                file_name='fold_test', 
                timestamp=True, 
                suffix=None, 
                tb_server=False, 
                adding_attributes=None):
        # super(self, FoldMetricBase).__init__(adding_attributes)
        self._fold_metric_saver = []
        self._cur_k = 0
        self._fold_acc = []
        self._strict_fold_acc = []
        self._k_fold = k_fold

        self.timestamp = time.strftime('%Y%m%d-%H_%M_%S', time.localtime(time.time()))
        self.file_name = file_name
        self.suffix = suffix
        self.dir_name = dir_name

        # self._base_saver = saver(adding_attributes)
        for k in range(k_fold):
            # self._fold_metric_saver.append(copy.deepcopy(self._base_saver))
            assert saver.__name__ in ['TensorBoardSaver', 'MetricSaverBase'], "your saver is not register in the FoldMetricBase, check the version or your spell"
            if saver.__name__ == 'TensorBoardSaver':
                self.tb = tb_server
                self._fold_metric_saver.append(saver(log_dir=os.path.join(dir_name, file_name, 'runs'), 
                                                    file_name=self.timestamp, 
                                                    tag=f'fold_{k}'))
                if k == 0:
                    self._fold_metric_saver[0].start(self.tb)
                    # self._fold_metric_saver[0].start_server()
            elif saver.__name__ == 'MetricSaverBase':
                self._fold_metric_saver.append(saver(adding_attributes=adding_attributes))
    
    def gen_path(self, dir_name, file_name, timestamp, suffix):
        if timestamp:
            # _time = time.strftime('%Y%m%d-%H:%M:%S', time.localtime(time.time()))
            _time = self.timestamp
        else:
            _time = ""

        if suffix is None:
            if len(self._strict_fold_acc):
                suffix = '{:.4f}-{:.4f}'.format(self.strict_mean_acc, self.mean_acc)
            else:
                suffix = '{:.4f}'.format(self.mean_acc)

        dir_name = os.path.join(dir_name, file_name, file_name+'-'+suffix+'-'+_time)
        return dir_name

    @property    
    def mean_acc(self):
        if len(self._fold_acc) == 0:
            return 0.00
        else:
            return np.mean(self._fold_acc)
    
    @property
    def strict_mean_acc(self):
        if len(self._strict_fold_acc) == 0:
            return 0.00
        else:
            return np.mean(self._strict_fold_acc)

    @property
    def cur_k(self):
        return min(len(self._fold_acc), self._cur_k)
    
    @property
    def std(self):
        if len(self._fold_acc) == 0:
            return 0.00
        else:
            return np.std(self._fold_acc)

    @property
    def strict_std(self):
        if len(self._strict_fold_acc) == 0:
            return 0.00
        else:
            return np.std(self._strict_fold_acc)
    
    @property
    def cur_saver(self):
        return self._fold_metric_saver[self._cur_k]
    
    @property
    def best_acc(self):
        if len(self._fold_acc) == 0:
            return 0.00
        else:
            return np.max(self._fold_acc)
    
    @property
    def cur_best_acc(self):
        return self.cur_saver.best_acc
    
    @property
    def cur_train_acc(self):
        return self.cur_saver.train_acc
    
    @property
    def cur_eva_acc(self):
        return self.cur_saver.eva_acc
    
    @property
    def cur_train_loss(self):
        return self.cur_saver.train_loss
    
    @property
    def cur_eva_loss(self):
        return self.cur_saver.eva_loss
    
    @property
    def cur_test_acc(self):
        return self.cur_saver.test_acc
    
    @property
    def cur_test_loss(self):
        return self.cur_saver.test_loss
    
    @property
    def list(self):
        return self._fold_acc

    @property
    def list_acc(self):
        return self._fold_acc

    def next_fold(self):
        self._fold_acc.append(self._fold_metric_saver[self._cur_k].best_acc)

        if self._fold_metric_saver[self._cur_k].has_test:
            self._strict_fold_acc.append(self._fold_metric_saver[self._cur_k].strict_best_acc)

        if self._cur_k + 1 >= self._k_fold:
            self._fold_metric_saver[self._cur_k].close(close_tb=True)
            return
        self._fold_metric_saver[self._cur_k].close()
        # 保存当前fold的acc
        # 指向当前fold的指针加1
        self._cur_k += 1
        self._fold_metric_saver[self._cur_k].start()
        # 若运行时制定的fold大于init时指定的fold，增加metric_saver的数量以使得两者匹配，防止因为metricsaver的错误导致程序的中断
        # if self._cur_k >= self._k_fold:
        #     self._k_fold = self._cur_k + 1
        # for i in range(self.cur_k, self._k_fold):
        #     self._fold_metric_saver.append(copy.deepcopy(self._base_saver))
    
    def add_record(self, *args, **kwargs):
        return self.cur_saver.add_record(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.add_record(*args, **kwargs)
    
    def EarlyStopping(self, *args, **kwargs):
        return self.cur_saver.EarlyStopping(*args, **kwargs)
    
    def close(self):
        pass

    def save(self):
        self.close()
        dir_name = self.gen_path(self.dir_name, self.file_name, self.timestamp, self.suffix)
        for i in range(self._k_fold):
            self._fold_metric_saver[i].save(dir_name, prefix=f'fold_{i}: ')

        if len(self._strict_fold_acc):
            ans = {
            'strict_mean_acc': self.strict_mean_acc,
            'strict_std': self.strict_std,
            'strict_list': self._strict_fold_acc,
            'mean_acc': self.mean_acc,
            'std': self.std,
            'list': self._fold_acc
            }
            with open('{}/{:.4f}-{:.4f}.json'.format(dir_name, self.strict_mean_acc, self.mean_acc), 'w') as f:
                json.dump(ans, f)
        else:
            ans = {
            'mean_acc': self.mean_acc,
            'std': self.std,
            'list': self._fold_acc
            }
            with open('{}/{:.4f}.json'.format(dir_name, self.mean_acc), 'w') as f:
                json.dump(ans, f)
        return f'save to {self.dir_name}'

    def __repr__(self):
        return f"Fold:{self._cur_k}/{self._k_fold} ||" + self._fold_metric_saver[self._cur_k].__repr__() + " || fold_mean_acc:{:.4f}, fold_best_acc:{:.4f}".format(self.mean_acc, self.best_acc)

if __name__ == "__main__":
    # a = MetricBase(adding_attributes=['x'])
    # fold = 10
    fold = 10
    # b = FoldMetricBase(k_fold=fold, adding_attributes=['x'])
    # b = FoldMetricBase(k_fold=fold, saver=TensorBoardSaver, dir_name='test_fold', file_name='test_tensorboard', adding_attributes=['x'], tb_server=True)
    b = FoldMetricBase(k_fold=fold, 
                       saver=TensorBoardSaver, 
                       dir_name='test_fold', 
                       file_name='test_tensorboard', 
                       adding_attributes=['x'], 
                       tb_server=False)
    for _ in range(fold):
        for i in range(100):
            _a, _b, _c, _d, _e, _f = np.random.rand(6)
            if b(_a, _b, _c, _d, _e, _f):
                print(_, i, _d, _e)
            # time.sleep(0.1)
            # if b.EarlyStopping(warmup=50, patience=20, min_delta=0.001):
            #     print(_, i, 'continue !!!')
            #     break
            # time.sleep(1)
        b.next_fold()
    b.save()

