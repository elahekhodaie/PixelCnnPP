import torch
import pcnnpp.config as config
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import time
from pcnnpp.data import DatasetSelection
from pcnnpp.utils.functions import get_loss_function, get_hitmap_function

np.seterr(divide='ignore', invalid='ignore')


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    one_hot = one_hot.to(config.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def sample_from_discretized_mix_logistic_1d(l, nr_mix=config.nr_logistic_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)

    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1]  # [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # for mean, scale
    l = l.to(config.device)
    logit_probs = logit_probs.to(config.device)
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    temp = temp.to(config.device)
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    u = torch.FloatTensor(means.size())
    u.uniform_(1e-5, 1. - 1e-5)
    u = u.to(config.device)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    out = x0.unsqueeze(1)
    return out


def sample_from_discretized_mix_logistic(l, nr_mix=config.nr_logistic_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    temp = temp.to(config.device)
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    coeffs = torch.sum(torch.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    u.uniform_(1e-5, 1. - 1e-5)
    u = u.to(config.device)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    x1 = torch.clamp(torch.clamp(
        x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    x2 = torch.clamp(torch.clamp(
        x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out


def get_sampler_function(input_shape):
    if input_shape[2] == 3:
        return sample_from_discretized_mix_logistic
    return sample_from_discretized_mix_logistic_1d


def sample(model, input_shape):
    model.train(False)
    data = torch.zeros(config.sample_batch_size, input_shape[0], input_shape[1], input_shape[2], device=config.device)
    for i in range(input_shape[1]):
        for j in range(input_shape[2]):
            data_v = data
            out = model(data_v, sample=True).to(config.device)
            out_sample = get_sampler_function(input_shape)(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data


def evaluate(model, dataset_test=None, test_dataloader=None, batch_size=config.test_batch_size,
             positive_is_anomaly=True):
    if dataset_test is None:
        print('loading test data')
        dataset_test = DatasetSelection(dataset=config.test_dataset, train=False, classes=config.test_classes)
    if test_dataloader is None:
        print('initializing test data loader')
        test_dataloader = dataset_test.get_dataloader(batch_size=batch_size, shuffle=False)
    input_shape = dataset_test.input_shape()
    hitmap_function = get_hitmap_function(input_shape)

    with torch.no_grad():
        data = None
        print(f'processing {len(test_dataloader)} test batches:')
        time_ = time.time()
        for idx, (inputs, labels) in enumerate(test_dataloader):
            images = inputs.numpy()
            inputs = inputs.to(config.device)
            output = model(inputs)
            hitmap = hitmap_function(inputs, output).cpu().numpy()
            log_prob = hitmap.sum(1).sum(1)
            is_anomaly = (np.isin(labels.numpy(), config.normal_classes) == (not positive_is_anomaly))
            results = np.array(
                [(is_anomaly[i], images[i].reshape(input_shape[1], input_shape[2]), hitmap[i], log_prob[i]) for i in
                 range(len(images))])
            data = results if data is None else np.append(data, results, axis=0)
            if (idx + 1) % config.evaluate_print_every == 0 and config.evaluate_print_every:
                print(
                    '\t{:3d}/{:3d} - time : {:.3f}s'.format(
                        idx + 1,
                        len(test_dataloader),
                        time.time() - time_)
                )
                time_ = time.time()
    return data


def plot_evaluation(data: np.array, model_name=config.model_name, positive_is_anomaly=True, save_path=None):
    # constant values
    line_width = 2
    title_font_size = 14

    # calculating roc curve
    fpr, tpr, thr = roc_curve(y_true=data[:, 0].astype(int), y_score=data[:, 3], pos_label=1)
    auc_score = roc_auc_score(y_true=data[:, 0].astype(int), y_score=data[:, 3])

    fig, axs = plt.subplots(2, 1, figsize=(9, 9.5))
    # roc
    fig.suptitle(f'{model_name} Evaluation (positive={"Anomaly" if positive_is_anomaly else "Normal"})',
                 fontsize=title_font_size)
    axs[0].plot(fpr, tpr, color='darkorange',
                lw=line_width, label='ROC curve (area = %0.3f)' % auc_score)
    axs[0].plot([0, 1], [0, 1], color='navy', lw=line_width / 2, linestyle='--')
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.01])
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title(f'ROC - {model_name}')
    axs[0].legend(loc="lower right")
    axs[0].set_aspect('equal')

    # precision
    tp = tpr * np.count_nonzero(data[:, 0])
    fp = fpr * np.count_nonzero(data[:, 0] == False)
    axs[1].plot(thr, tp / (tp + fp), lw=line_width, color='darkorange')
    axs[1].plot([thr.min(), thr.max()], [0.5, 0.5], lw=line_width / 2, color='navy', linestyle='--')
    axs[1].set_xlim(thr.min(), thr.max())
    axs[1].set_ylim([0.0, 1.01])
    axs[1].set_xlabel('Threshold on Log(Likelihood)')
    axs[1].set_ylabel('Precision')
    axs[1].set_title('Precision')
    axs[1].set_aspect('auto')
    if save_path is not None:
        fig.savefig(save_path)
    fig.show()
    plt.clf()
    plt.cla()


def plot_loss(training_loss, validation_loss, model_name=config.model_name, save_path=None):
    # loss tracking
    if len(training_loss) != len(validation_loss):
        return
    plt.plot(range(config.start_epoch, config.start_epoch + len(validation_loss)), validation_loss, label='Validation')
    plt.plot(range(config.start_epoch, config.start_epoch + len(validation_loss)), training_loss, label='Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(config.start_epoch - 1, config.start_epoch + len(validation_loss) + 1)
    plt.title(f'{model_name} Epoch Losses {config.start_epoch + len(validation_loss) - 1}')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.clf()
