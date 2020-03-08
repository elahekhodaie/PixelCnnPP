import torch
import pcnnpp.config as config
from pcnnpp.data import DatasetSelection
from pcnnpp.utils.functions import log_prob_from_logits
import matplotlib.pyplot as plt


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


def evaluate_roc(model):
    print('loading test data')
    dataset_test = DatasetSelection(dataset=config.test_dataset, train=False, classes=config.normal_classes)
    test_dataloader = dataset_test.get_dataloader(batch_size=config.test_batch_size, shuffle=not config.use_tpu)
    model = model.to(config.device)
    all_outputs = torch.empty((1,))
    all_labels = torch.empty((1,))
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            output = model(inputs)
            log_prob_from_logits(output)
            outputs = torch.cat((outputs, log_prob_from_logits(output)), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)
        outputs.sum(dim=4)

def plot_loss(training_loss, validation_loss):
    if len(training_loss) != len(validation_loss):
        return
    plt.plot(range(config.start_epoch, config.start_epoch + len(validation_loss)), validation_loss, label='Validation')
    plt.plot(range(config.start_epoch, config.start_epoch + len(validation_loss)), training_loss, label='Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Epoch Losses {config.start_epoch + len(validation_loss) - 1}')
    plt.legend()
    plt.show()
