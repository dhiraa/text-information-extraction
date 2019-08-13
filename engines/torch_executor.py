import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TorchExecutor(object):
    def __init__(self):
        cudnn.benchmark = True
        cudnn.deterministic = True
        num_gpu = torch.cuda.device_count()
        # print('device count', opt.num_gpu)
        if num_gpu > 1:
            print('------ Use multi-GPU setting ------')
            print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
            # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
            workers = workers * num_gpu
