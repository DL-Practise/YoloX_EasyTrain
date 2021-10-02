import sys
import os
import shutil
import subprocess
import time
import multiprocessing
import threading
import psutil

def train_func(train_cmd,  prj):
    print('-------------------> train thread start...')
    log_file = prj.log_file
    info_cb = prj.train_cb
    prj.train_subprocess = subprocess.Popen(train_cmd, stdout=None, stderr=None) #, close_fds=True)
    proces_info = psutil.Process(prj.train_subprocess.pid)
    
    #wait for log_file create
    while not os.path.exists(log_file):
        time.sleep(1)
        print('train thread waiting for log file create.')

    with open(log_file, 'r+') as f:
        while True :
            prj.train_status = proces_info.status()
            if not prj.is_train():
                break
                
            line = f.readline()
            if line and info_cb is not None:
                #print('>>>', line.strip('\r\n'))
                info_cb({'log': line.strip('\r\n')})
                info = prj.phase_log_line(line)
                if info is not None:
                    info_cb(info)
            else:
                time.sleep(1)
    print('-------------------> train thread stop...')
    info_cb({'train_status': 'finished'})

class ZXProj():
    def __init__(self, prj_name):
        self.prj_name = prj_name
        self.root_dir = './projects/'
        self.prj_dir = self.root_dir + self.prj_name
        self.exp_file = self.root_dir + self.prj_name + "/%s_exp.py"%self.prj_name
        self.class_file = self.root_dir + self.prj_name + "/%s_classes.py"%self.prj_name
        self.outputs_dir = self.root_dir + self.prj_name + "/outputs"
        self.model_dir = self.root_dir + self.prj_name + "/outputs/%s_exp/"%self.prj_name
        self.log_file = self.root_dir + self.prj_name + "/outputs/%s_exp/train_log.txt"%self.prj_name
        self.vis_dir = self.root_dir + self.prj_name + "/outputs/%s_exp/vis_res/"%self.prj_name
        self.exp_cache = []
        
        self.train_process = None
        self.train_subprocess = None
        self.train_cb = None
        self.train_status = 'unknown'

        self.accu_info = {'epoch': 0, 'map': [0.0, 0.0, 0.0]}

        #self.yolox_root = '/data/zhengxing/my_dl/train/YOLOX-main/' #############??????
        self.create_proj()

    def create_proj(self):
        if not os.path.exists(self.prj_dir):
            print("prj: ", self.prj_dir, " is not exist. create it")
            os.makedirs(self.prj_dir)
            os.makedirs(self.outputs_dir)
            shutil.copy('backup/backup_exp.py', self.exp_file)
            shutil.copy('backup/backup_classes.py', self.class_file)
        else:
            print("prj: ", self.prj_dir, " is already exist. give up create")
        self._load_exp_file_to_cache()
        self._modify_exp_cache("output_dir", '\"' + self.outputs_dir + '\"')
    
    def delete_proj(self):
        if os.path.exists(self.prj_dir):
            shutil.rmtree(self.prj_dir)
           
    def reset_exp(self):
        if not os.path.exists(self.prj_dir):
            print("prj: ", self.prj_dir, " is not exist. give up reset exp")
        else:
            print("prj: ", self.prj_dir, " is exist. start to get exp")
            shutil.copy('backup/backup_exp.py', self.exp_file)
        self._load_exp_file_to_cache()
        self._modify_exp_cache("output_dir", self.outputs_dir)

    def get_exp(self):
        IGNORE_KEYS = ['output_dir', 'exp_name']
        now_key = None
        config_map = {}
        for line in self.exp_cache:
            
            if line.strip(' ').startswith('#') and '---' in line and 'config' in line:
                now_key = line.replace('#','').replace('-', '').lstrip().rstrip()
                if now_key not in config_map.keys():
                    config_map[now_key] = {}
            if not line.strip(' ').startswith('#') and 'self' in line and '=' in line and now_key is not None:
                fileds = line.strip().split('=')
                param_name = fileds[0].replace('self.', '').lstrip().rstrip()
                if param_name in IGNORE_KEYS:
                    continue
                param_value = fileds[1].lstrip().rstrip()
                #print(param_name, param_value)
                config_map[now_key][param_name] = param_value

        return config_map
       
    def save_exp(self, exp_map):
        for key in exp_map.keys():
            for sub_key in exp_map[key].keys():
                self._modify_exp_cache(sub_key, exp_map[key][sub_key])
        with open(self.exp_file, "w") as f:
            for line in self.exp_cache:
                f.writelines(line + '\n')

    def start_train(self, gpu_num, batch_size, mix=False, pretrain='', info_cb=None):
        if self.train_process is None:
            if os.path.exists(self.log_file):
                shutil.copy(self.log_file, self.log_file + '_' + str(time.time()))
                with open(self.log_file, 'w') as f:
                    f.seek(0)
                    f.truncate()
            
            train_cmd = [str(sys.executable), self._get_exp_value("yolox_root").replace('"', '') + "/tools/train.py", 
                         "-f", self.exp_file, 
                         "-d", gpu_num, 
                         "-b", batch_size]

            if mix:
                train_cmd.append('--fp16')
            if pretrain != '':
                train_cmd.append('-o')
                train_cmd.append('-c')
                train_cmd.append(pretrain)
            if info_cb is not None:
                self.train_cb = info_cb

            #self.train_process = multiprocessing.Process(target=train_func, args=(train_cmd, self,)) 
            self.train_process = threading.Thread(target=train_func, args=(train_cmd, self,)) 
            self.train_process.start()

    def stop_train(self):
        if self.train_process is not None:
            if self.train_subprocess is not None:
                try:
                    self.train_subprocess.kill()
                except:
                    pass
                self.train_subprocess = None
            try:
                self.train_process.join()
            except:
                pass
            self.train_process = None

    def is_train(self):
        #print('***** status: ', self.train_status)
        if 'running' in self.train_status or 'sleep' in self.train_status:
            return True
        else:
            return False

    def get_log_info(self, full=False):
        def _get_last_line(file_path):
            with open(file_path, 'rb') as f:
                first_line = f.readline()
                offset = -50
                while True:
                    f.seek(offset, 2)
                    lines = f.readlines()
                    if len(lines) >= 2:
                        last_line = lines[-1]
                        break
                    offset *= 2
            return first_line.decode(), last_line.decode()

        def _phase_one_line(line):
            if 'epoch:' in line and 'iter:' in line and 'total_loss:' in line:
                fileds = line.split(',')
                epoch_info = int(fileds[0].split('-')[-1].split(':')[-1].split('/')[0])
                total_epoch = int(fileds[0].split('-')[-1].split(':')[-1].split('/')[-1])
                iter_per_epoch = int(fileds[1].split(':')[-1].split('/')[1])
                iter_info = int(fileds[1].split(':')[-1].split('/')[0]) + (epoch_info - 1) * iter_per_epoch
                mem_info = float(fileds[2].split(':')[-1].split('M')[0])
                iter_time = float(fileds[3].split(':')[-1].split('s')[0])
                total_loss = float(fileds[5].split(':')[-1])
                iou_loss = float(fileds[6].split(':')[-1])
                l1_loss = float(fileds[7].split(':')[-1])
                conf_loss = float(fileds[8].split(':')[-1])
                cls_loss = float(fileds[9].split(':')[-1])
                lr = float(fileds[10].split(':')[-1])
                progress = float(epoch_info) / float(total_epoch)
                remain_time = fileds[-1].split(' ')[-1].strip('\r\n')
                return [progress, remain_time, iter_info, total_loss, iou_loss, l1_loss, conf_loss, cls_loss, lr] 
            else:
                return []
            
        if full:
            full_data = [[],[],[],[],[],[],[],[],[]]
            try:
                if os.path.exists(self.log_file):
                    with open(self.log_file, 'r') as f:
                        for line in f.readlines():
                            
                            line = line.strip()
                            info_list = _phase_one_line(line)
                        
                            if len(info_list) > 0:
                                #print('###', len(info_list))
                                assert(len(full_data) == len(info_list))
                                for i in range(len(full_data)):
                                    full_data[i].append(info_list[i])
            except:
                full_data = [[],[],[],[],[],[],[],[],[]]
            return  full_data      
        else:
            add_data = []
            try:
                if os.path.exists(self.log_file):
                    _, last_line = _get_last_line(self.log_file)
                    add_data = _phase_one_line(last_line)
            except:
                add_data = []
            return add_data
    
    def phase_log_line(self, line):
        if 'epoch:' in line and 'iter:' in line and 'total_loss:' in line:
            fileds = line.split(',')
            epoch_info = int(fileds[0].split('-')[-1].split(':')[-1].split('/')[0])
            self.accu_info['epoch'] = epoch_info
            total_epoch = int(fileds[0].split('-')[-1].split(':')[-1].split('/')[-1])
            iter_per_epoch = int(fileds[1].split(':')[-1].split('/')[1])
            iter_in_epoch = int(fileds[1].split(':')[-1].split('/')[0])
            iter_info = iter_in_epoch + (epoch_info - 1) * iter_per_epoch
            mem_info = float(fileds[2].split(':')[-1].split('M')[0])
            iter_time = float(fileds[3].split(':')[-1].split('s')[0])
            total_loss = float(fileds[5].split(':')[-1])
            iou_loss = float(fileds[6].split(':')[-1])
            l1_loss = float(fileds[7].split(':')[-1])
            conf_loss = float(fileds[8].split(':')[-1])
            cls_loss = float(fileds[9].split(':')[-1])
            lr = float(fileds[10].split(':')[-1])
            progress_value = float(epoch_info) / float(total_epoch)
            progress_str = 'epoch[%d/%d] iter[%d/%d]'%(epoch_info,total_epoch,iter_in_epoch,iter_per_epoch)
            remain_time = fileds[-1].split(' ')[-1].strip('\r\n')

            return {'iter_info': iter_info,
                    'loss_info': [total_loss, iou_loss, l1_loss, conf_loss, cls_loss], 
                    'lr_info': lr,
                    'progress_info': [progress_value, iter_time, progress_str, remain_time]}
        elif 'Average Precision' in line and 'IoU=0.50:0.95' in line:
            fileds = line.split('=')
            self.accu_info['map'][2] = float(fileds[-1])
            return None
        elif 'Average Precision' in line and 'IoU=0.50  ' in line:
            fileds = line.split('=')
            self.accu_info['map'][0] = float(fileds[-1])
            return None
        elif 'Average Precision' in line and 'IoU=0.75  ' in line:
            fileds = line.split('=')
            self.accu_info['map'][1] = float(fileds[-1])
            return self.accu_info
        else:
            return None

    def infer_img(self, model_path, img_path, test_conf, nms_conf, size, result_cb):
        infer_cmd = [str(sys.executable), self._get_exp_value("yolox_root").replace('"', '') + "/tools/demo.py", 
                     "image",
                     "-f", self.exp_file, 
                     "-c", model_path,
                     "--path", img_path, 
                     "--conf", test_conf,
                     "--nms", nms_conf,
                     "--tsize", size,
                     "--save_result",
                     "--device", 'cpu']

        print('infer start')
        p = subprocess.Popen(infer_cmd, stdout=None, stderr=None)
        p.wait()
        print('infer finished')

        last_dir='0'
        for sub_dir in os.listdir(self.vis_dir):
            print('sub_dir: ', sub_dir)
            if str(sub_dir) > last_dir:
                last_dir = sub_dir
        last_dir = os.path.join(self.vis_dir , last_dir)

        for file in os.listdir(last_dir):
            last_file = os.path.join(last_dir, str(file))
            result_cb(last_file)
            break

    def _get_exp_value(self, key):
        for line in self.exp_cache:
            if not line.strip(' ').startswith('#') and 'self' in line and '=' in line:
                fileds = line.strip().split('=')
                param_name = fileds[0].replace('self.', '').lstrip().rstrip()
                if param_name == key:
                    param_value = fileds[1].lstrip().rstrip()
                    return param_value
        return None

    def _load_exp_file_to_cache(self):
        print('ZXProj::_load_exp_file_to_cache: exp_file is:',self.exp_file)
        self.exp_cache = []
        with open(self.exp_file, "r") as f:
            for line in f.readlines():
                self.exp_cache.append(line.strip('\r\n'))
                
    def _show_exp_cache(self):
        for line in self.exp_cache:
            print(line)
            
    def _modify_exp_cache(self, key, value):
        line_index = -1
        new_line = ""
        for i, line in enumerate(self.exp_cache):
            if not line.strip(' ').startswith('#') and key in line and "self" in line and "=" in line:
                line_index = i
                fields = line.split('=')
                new_line = fields[0] + "= " + value
                break
        self.exp_cache[line_index] = new_line


if __name__ == '__main__':
    prj = ZXProj("xxx")
    prj.create_proj()