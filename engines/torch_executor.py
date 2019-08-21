import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from dataset.scene_text_recognition.utils import Averager
from engines.executor_base import ExecutorBase
from print_helper import print_info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TorchExecutor(ExecutorBase):
    def __init__(self,
                 experiment_name,
                 model,
                 dataset,
                 max_train_steps,
                 validation_interval_steps,
                 workers=4):
        
        ExecutorBase.__init__(self,
                              experiment_name=experiment_name,
                              model=model,
                              dataset=dataset,
                              data_iterator=None,
                              max_train_steps=max_train_steps,
                              validation_interval_steps=validation_interval_steps)
        
        assert (isinstance(model, torch.nn.Module))
        self._experiment_name = experiment_name
        self._model = model
        self._dataset = dataset
        self._validation_interval_steps = validation_interval_steps

        cudnn.benchmark = True
        cudnn.deterministic = True
        num_gpu = torch.cuda.device_count()
        # print('device count', opt.num_gpu)
        if num_gpu > 1:
            print('------ Use multi-GPU setting ------')
            print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
            # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
            workers = workers * num_gpu

        # weight initialization
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue

    def store_model(self, model, file_name):

        if not os.path.exists(f"./sore/{self._experiment_name}/{model.__class__.__name__}"):
            os.makedirs(f"./sore/{self._experiment_name}/{model.__class__.__name__}")

        torch.save(model, f"./sore/{self._experiment_name}/{model.__class__.__name__}/{file_name}")

    def validation(self, model):
        """ validation or evaluation """
        criterion = self._model.get_loss_op()
        evaluation_loader = self._dataset.validation_set()
        converter = self._model.get_converter()
        n_correct = 0
        norm_ED = 0
        length_of_data = 0
        infer_time = 0
        valid_loss_avg = Averager()
        batch_max_length = 25

        for i, (image_tensors, labels) in enumerate(evaluation_loader):
            batch_size = image_tensors.size(0)
            length_of_data = length_of_data + batch_size
            image = image_tensors.to(device)

            forward_time, cost, preds_str = self._model.get_predictions(model=model,
                                                                        batch_size=self._dataset._batch_size,
                                                                        image=image,
                                                                        labels=labels)
            infer_time += forward_time
            valid_loss_avg.add(cost)

            n_correct_, norm_ED_ = self._model.get_accuracy(preds_str=preds_str, labels=labels)
            n_correct += n_correct_
            norm_ED += norm_ED_

        accuracy = n_correct / float(length_of_data) * 100

        return valid_loss_avg.val(), accuracy, norm_ED, preds_str, labels, infer_time, length_of_data

    def train(self, num_max_steps=None, num_epoch=None):
        assert (num_max_steps is not None and num_epoch is not None, "Use steps or epoch at a time")
        model_dir = self._model.model_dir
        # data parallel for multi-GPU
        model = torch.nn.DataParallel(self._model).to(device)
        model.train()

        num_samples = len(self._dataset)
        batch_size = self._dataset._batch_size

        num_steps_per_epoch = num_samples // batch_size

        current_step = 0
        i = 0
        total_num_steps = -1

        if num_epoch:
            total_num_steps = num_steps_per_epoch * num_epoch

        if num_max_steps:
            total_num_steps = num_max_steps

        # loss averager
        loss_avg = Averager()

        train_dataset = self._dataset.get_train_dataset()
        converter = self._model.get_converter()
        # criterion = self._model.get_loss_op()

        start_time = time.time()
        best_accuracy = -1
        best_norm_ED = 1e+6

        while (current_step < total_num_steps):
            # train part
            image_tensors, labels = train_dataset.get_batch()
            image = image_tensors.to(device)
            print_info(labels)
            text, length = converter.encode(labels, batch_max_length=25) #self.batch_max_length)
            batch_size = image.size(0)

            feature = dict()
            label = dict()
            feature["image"], label["text"], label["length"] = image, text, length
            cost = self._model.get_cost(model=self._model, feature=feature, label=label)

            optimizer = self._model.get_optimizer(model=model)

            model.zero_grad()
            cost.backward()
            grad_clip = 5
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # gradient clipping with 5 (Default)
            optimizer.step()

            loss_avg.add(cost)

            # validation part
            if i % self._validation_interval_steps == 0:
                elapsed_time = time.time() - start_time
                print(f'[{i}/{self._max_train_steps}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f}')
                # for log
                with open(f'./store/{self._experiment_name}/log_train.txt', 'a') as log:
                    log.write(f'[{i}/{self._max_train_steps}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f}\n')
                    loss_avg.reset()

                    model.eval()
                    with torch.no_grad():
                        valid_loss, current_accuracy, current_norm_ED, \
                        preds, labels, infer_time, length_of_data = self.validation(model=model)
                    model.train()

                    for pred, gt in zip(preds[:5], labels[:5]):
                        if 'Attn' in opt.Prediction:
                            pred = pred[:pred.find('[s]')]
                            gt = gt[:gt.find('[s]')]
                        print(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}')
                        log.write(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}\n')

                    valid_log = f'[{i}/{self._max_train_steps}] valid loss: {valid_loss:0.5f}'
                    valid_log += f' accuracy: {current_accuracy:0.3f}, norm_ED: {current_norm_ED:0.2f}'
                    print(valid_log)
                    log.write(valid_log + '\n')

                    # keep best accuracy model
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        torch.save(model.state_dict(), f'./saved_models/{self._experiment_name}/best_accuracy.pth')
                    if current_norm_ED < best_norm_ED:
                        best_norm_ED = current_norm_ED
                        torch.save(model.state_dict(), f'./saved_models/{self._experiment_name}/best_norm_ED.pth')
                    best_model_log = f'best_accuracy: {best_accuracy:0.3f}, best_norm_ED: {best_norm_ED:0.2f}'
                    print(best_model_log)
                    log.write(best_model_log + '\n')

            # save model per 1e+5 iter.
            if (i + 1) % 1e+5 == 0:
                torch.save(model.state_dict(), f'./saved_models/{self._experiment_name}/iter_{i + 1}.pth')

            if i == self._max_train_steps:
                print('end the training')
                sys.exit()

            i += 1
            current_step += 1
