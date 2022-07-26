import os

import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import os.path as osp

from device import device
from loss import compute_center_loss, get_center_delta, compute_relative_loss


class Trainer(object):

    def __init__(
            self, optimizer, model, training_dataloader,
            validation_dataloader, log_dir=False, max_epoch=100, resume=False,
            persist_stride=5, lamda=0.03, alpha=0.5):

        self.log_dir = log_dir
        self.optimizer = optimizer
        self.model = model
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

        if not self.log_dir:
            self.log_dir = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), 'logs')
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        if resume:
            state_file = os.path.join(self.log_dir, 'models', resume)
            if not os.path.isfile(state_file):
                raise RuntimeError(
                    "resume file {} is not found".format(state_file))
            print("loading checkpoint {}".format(state_file))
            checkpoint = torch.load(state_file)
            self.start_epoch = self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
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

        soft_feat = np.load('datasets/lfw/atts_lfw.npy')
        soft_feat = torch.tensor(soft_feat.astype('float32'))
        soft_feat = soft_feat.to(device)

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
                #print(features.shape)

                cross_entropy_loss = torch.nn.functional.cross_entropy(
                    logits, targets)
                center_loss = compute_center_loss(features, centers, targets)
                relative_loss = compute_relative_loss(features, centers, targets, soft_feat)
                center_loss = relative_loss
                loss = cross_entropy_loss + center_loss + relative_loss*0.5

                if (batch + 1) % 20 == 0:
                    #print(centers)
                    print("[{}:{}] cross entropy loss: {:.8f} - center loss: "
                          "{:.8f} - total weighted loss: {:.8f}".format(
                              mode, self.current_epoch,
                              cross_entropy_loss.item(),
                              center_loss.item(), loss.item()))

                total_cross_entropy_loss += cross_entropy_loss
                total_center_loss += center_loss
                total_loss += loss

                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # make features untrack by autograd, or there will be
                    # a memory leak when updating the centers
                    center_deltas = get_center_delta(
                        features.data, centers, targets, self.alpha)
                    self.model.centers = centers - center_deltas

                # compute acc here
                total_top1_matches += self._get_matches(targets, logits, 1)
                total_top3_matches += self._get_matches(targets, logits, 3)

            all_features, all_labels, all_features_2d = self.get_dataset_feat(dataloader, epoch=self.current_epoch, prefix=mode)
            self.plot_features(all_features, epoch=self.current_epoch, prefix=mode)
            #self.plot_features(all_features_2d, all_labels, num_classes=centers.size(0), epoch=self.current_epoch, prefix=mode)



            center_loss = total_center_loss / batch
            cross_entropy_loss = total_cross_entropy_loss / batch
            loss = center_loss + cross_entropy_loss
            top1_acc = total_top1_matches / len(dataloader.dataset)
            top3_acc = total_top3_matches / len(dataloader.dataset)

            loss_recorder['center'].append(total_center_loss/batch)
            loss_recorder['cross_entropy'].append(cross_entropy_loss)
            loss_recorder['together'].append(total_loss/batch)
            loss_recorder['top1acc'].append(top1_acc)
            loss_recorder['top3acc'].append(top3_acc)

            print(
                "[{}:{}] finished. cross entropy loss: {:.8f} - "
                "center loss: {:.8f} - together: {:.8f} - "
                "top1 acc: {:.4f} % - top3 acc: {:.4f} %".format(
                    mode, self.current_epoch, cross_entropy_loss.item(),
                    center_loss.item(), loss.item(),
                    top1_acc*100, top3_acc*100))



    def _get_matches(self, targets, logits, n=1):
        _, preds = logits.topk(n, dim=1)
        targets_repeated = targets.view(-1, 1).repeat(1, n)
        matches = torch.sum(preds == targets_repeated, dim=1) \
            .nonzero().size()[0]
        return matches

    def persist(self, is_best=False):
        model_dir = os.path.join(self.log_dir, 'models')
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


    def get_dataset_feat(self, dataloader, epoch, prefix):

        all_features, all_labels = [], []
        batch = 0

        for images, targets, names in dataloader:

            batch = batch + 1

            targets = torch.tensor(targets).to(device)
            images = images.to(device)
            centers = self.model.centers

            logits, features = self.model(images)

            all_features.append(features.detach().cpu().numpy())
            all_labels.append(targets.detach().cpu().numpy())

        # convert to np.array
        all_features = np.vstack(all_features)
        all_features_2d = all_features
        #all_features_2d = TSNE(n_components=2, init = 'random').fit_transform(all_features)
        all_labels = np.hstack(all_labels)

        dirname = osp.join('logs', prefix)
        if not osp.exists(dirname):
            os.mkdir(dirname)
        save_name = osp.join(dirname, 'epoch_' + str(epoch + 1) + '.npz')
        np.savez(save_name, features=all_features, labels=all_labels)

        return all_features, all_labels, all_features_2d


    def plot_features(self, features, epoch, prefix):

        """Plot features on 2D plane.

                Args:
                    features: (num_instances, num_features).
                    labels: (num_instances).
                """

        plt.scatter(features[:, 0], features[:, 1])

        dirname = osp.join('logs', prefix)
        if not osp.exists(dirname):
            os.mkdir(dirname)
        save_name = osp.join(dirname, 'epoch_' + str(epoch + 1) + '.png')
        plt.savefig(save_name, bbox_inches='tight')
        # plt.show()
        plt.close()


    def plot_features_per_class(self, features, labels, num_classes, epoch, prefix):
        """Plot features on 2D plane.

        Args:
            features: (num_instances, num_features).
            labels: (num_instances).
        """
        colors = ['C' + str(i) for i in range(10)]
        for label_idx in range(10):#num_classes):
            plt.scatter(
                features[labels == label_idx, 0],
                features[labels == label_idx, 1],
                c=colors[label_idx],
                s=1,
            )
        plt_labels = [str(i) for i in range(10)]
        plt.legend(plt_labels, loc='upper right')
        dirname = osp.join('logs', prefix)
        if not osp.exists(dirname):
            os.mkdir(dirname)
        save_name = osp.join(dirname, 'epoch_' + str(epoch + 1) + '.png')
        plt.savefig(save_name, bbox_inches='tight')
        #plt.show()
        plt.close()
