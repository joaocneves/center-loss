import os

import torch

from device import device
from loss import compute_center_loss, get_center_delta

from eval import evaluate


class Trainer(object):

    def __init__(
            self, optimizer, model, loss, training_dataloader,
            validation_dataloader, test_set_path, log_dir=False, experiment_name='', max_epoch=100, resume=False,
            persist_stride=2, lamda=3, alpha=0.5):

        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.optimizer = optimizer
        self.model = model
        self.loss = loss
        self.max_epoch = max_epoch
        self.resume = resume
        self.persist_stride = persist_stride
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.training_losses = {
                'center': [], 'cross_entropy': [],
                'together': [], 'top3acc': [], 'top1acc': []}
        self.validation_losses = {
                'center': [], 'cross_entropy': [],
                'together': [], 'top3acc': [], 'top1acc': []}
        self.start_epoch = 1
        self.current_epoch = 1
        self.lamda = lamda
        self.alpha = alpha
        self.test_set_path = test_set_path

        if not self.log_dir:
            self.log_dir = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), 'logs')
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        if resume:
            state_file = os.path.join(self.log_dir, self.experiment_name, resume)
            if not os.path.isfile(state_file):
                raise RuntimeError(
                    "resume file {} is not found".format(state_file))
            print("loading checkpoint {}".format(state_file))
            checkpoint = torch.load(state_file)
            self.start_epoch = self.current_epoch = checkpoint['epoch']

            pretrained_dict = checkpoint['state_dict']
            #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('base')}

            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            #self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.training_losses = checkpoint['training_losses']
            self.validation_losses = checkpoint['validation_losses']
            print("loaded checkpoint {} (epoch {})".format(
                state_file, self.current_epoch))

    def train(self):
        for self.current_epoch in range(self.start_epoch, self.max_epoch+1):
            self.run_epoch(mode='train')
            self.run_epoch(mode='validate')
            if not (self.current_epoch % self.persist_stride):
                self.persist()

    def run_epoch(self, mode):
        if mode == 'train':
            dataloader = self.training_dataloader
            loss_recorder = self.training_losses
            self.model.train()
        else:
            dataloader = self.validation_dataloader
            loss_recorder = self.validation_losses
            self.model.eval()

        total_cross_entropy_loss = 0
        total_center_loss = 0
        total_loss = 0
        total_top1_matches = 0
        total_top3_matches = 0
        batch = 0


        with torch.set_grad_enabled(mode == 'train'):
            for images, targets, names in dataloader:
                batch += 1
                targets = torch.tensor(targets).to(device)
                images = images.to(device)
                centers = self.model.centers

                logits, features = self.model(images)

                cross_entropy_loss = torch.nn.functional.cross_entropy(
                    logits, targets)
                if self.loss == 'softmax':
                    loss = cross_entropy_loss
                elif self.loss == 'center-loss':
                    center_loss = compute_center_loss(features, centers, targets)
                    loss = self.lamda * center_loss + cross_entropy_loss
                elif self.loss == 'attribute-loss':
                    loss = self.lamda * center_loss + cross_entropy_loss
                
                if (batch + 1) % 500 == 0:
                    print("[{}:{}] cross entropy loss: {:.8f} - center loss: "
                          "{:.8f} - total weighted loss: {:.8f}".format(
                              mode, self.current_epoch,
                              cross_entropy_loss.item(),
                              center_loss.item(), loss.item()))


                total_cross_entropy_loss += cross_entropy_loss
                if self.loss == 'center-loss':
                    total_center_loss += center_loss
                total_loss += loss

                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if self.loss == 'center-loss':
                        # make features untrack by autograd, or there will be
                        # a memory leak when updating the centers
                        center_deltas = get_center_delta(
                            features.data, centers, targets, self.alpha)
                        self.model.centers = centers - center_deltas

                # compute acc here
                total_top1_matches += self._get_matches(targets, logits, 1)
                total_top3_matches += self._get_matches(targets, logits, 3)


            cross_entropy_loss = total_cross_entropy_loss / batch
            if self.loss == 'softmax':
                loss = cross_entropy_loss
            elif self.loss == 'center-loss':
                center_loss = total_center_loss / batch
                loss = center_loss + cross_entropy_loss
            top1_acc = total_top1_matches / len(dataloader.dataset)
            top3_acc = total_top3_matches / len(dataloader.dataset)

            if self.loss == 'center-loss':
                loss_recorder['center'].append(total_center_loss/batch)
            loss_recorder['cross_entropy'].append(cross_entropy_loss)
            loss_recorder['together'].append(total_loss/batch)
            loss_recorder['top1acc'].append(top1_acc)
            loss_recorder['top3acc'].append(top3_acc)

            if self.loss == 'softmax':
                print(
                    "[{}:{}] finished. cross entropy loss: {:.8f} - "
                    "top1 acc: {:.4f} % - top3 acc: {:.4f} %".format(
                        mode, self.current_epoch, cross_entropy_loss.item(),
                        top1_acc * 100, top3_acc * 100))
            elif self.loss == 'center-loss':
                print(
                    "[{}:{}] finished. cross entropy loss: {:.8f} - "
                    "center loss: {:.8f} - together: {:.8f} - "
                    "top1 acc: {:.4f} % - top3 acc: {:.4f} %".format(
                        mode, self.current_epoch, cross_entropy_loss.item(),
                        center_loss.item(), loss.item(),
                        top1_acc*100, top3_acc*100))

            evaluate(model=self.model, test_set_path=self.test_set_path, pairs_file='datasets/lfw/pairs.txt')

    def _get_matches(self, targets, logits, n=1):
        _, preds = logits.topk(n, dim=1)
        targets_repeated = targets.view(-1, 1).repeat(1, n)
        matches = torch.sum(preds == targets_repeated, dim=1) \
            .nonzero().size()[0]
        return matches

    def persist(self, is_best=False):
        model_dir = os.path.join(self.log_dir, self.experiment_name)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        file_name = (
            "epoch_{}_best.pth.tar" if is_best else "epoch_{}.pth.tar") \
            .format(self.current_epoch)

        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }
        state_path = os.path.join(model_dir, file_name)
        torch.save(state, state_path)
