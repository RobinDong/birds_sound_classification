import time
import datetime
import argparse

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn

from utils import augment

import pycls.core.builders as builders
from pycls.core.config import cfg

from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset.sounds import BirdsDataset, ListLoader


config = {
    "num_classes": 10958,
    "num_workers": 2,
    "save_folder": "ckpt/",
    "ckpt_name": "bird_cls",
}


def save_ckpt(net, iteration):
    torch.save(
        net.state_dict(),
        config["save_folder"]
        + config["ckpt_name"]
        + "_"
        + str(iteration)
        + ".pth",
    )


def evaluate(net, eval_loader):
    total_loss = 0.0
    batch_iterator = iter(eval_loader)
    sum_accuracy = 0.0
    sum_correct = 0
    eval_samples = 0
    aug = augment.Augment(training=False).cuda()
    for iteration in range(len(eval_loader)):
        sounds, type_ids = next(batch_iterator)
        if torch.cuda.is_available():
            sounds = Variable(sounds.cuda())
            type_ids = Variable(type_ids.cuda())

        # forward
        sounds = sounds.unsqueeze(3)
        sounds = aug(sounds)
        out = net(sounds.permute(0, 3, 1, 2).float())
        # accuracy
        _, predict = torch.max(out, 1)
        correct = (predict == type_ids)
        sum_accuracy += correct.sum().item() / correct.size()[0]
        sum_correct += correct.sum().item()
        eval_samples += sounds.shape[0]
        # loss
        loss = F.cross_entropy(out, type_ids)
        total_loss += loss.item()
    print("sum_correct:", sum_correct, eval_samples)
    print("sum_accuracy:", sum_accuracy, iteration)
    return total_loss / iteration, sum_correct / eval_samples


def warmup_learning_rate(optimizer, steps, warmup_steps):
    min_lr = args.lr / 100
    slope = (args.lr - min_lr) / warmup_steps

    lr = steps * slope + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train(args, train_loader, eval_loader):
    cfg.MODEL.TYPE = "regnet"
    cfg.REGNET.DEPTH = 25
    cfg.REGNET.SE_ON = False
    cfg.REGNET.W0 = 112
    cfg.REGNET.WA = 33.22
    cfg.REGNET.WM = 2.27
    cfg.REGNET.GROUP_W = 72
    cfg.BN.NUM_GROUPS = 4
    cfg.ANYNET.STEM_CHANNELS = 1
    cfg.MODEL.NUM_CLASSES = config["num_classes"]
    net = builders.build_model()
    net = net.cuda(device=torch.cuda.current_device())
    print("net", net)

    if args.resume:
        print("Resuming training, loading {}...".format(args.resume))
        ckpt_file = (
            config["save_folder"]
            + config["ckpt_name"]
            + "_"
            + str(args.resume)
            + ".pth"
        )
        net.load_state_dict(torch.load(ckpt_file))

    if args.finetune:
        print("Finetuning......")
        # Freeze all layers
        for param in net.parameters():
            param.requires_grad = False
        # Unfreeze some layers
        for layer in [net.s2.b10, net.s2.b11, net.s2.b12]:
            for param in layer.parameters():
                param.requies_grad = True
        net.head.fc.weight.requires_grad = True
        optimizer = optim.SGD(
            filter(lambda param: param.requires_grad, net.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            nesterov=False,
        )
    else:
        optimizer = optim.SGD(
            net.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            nesterov=False,
        )

    scheduler = ReduceLROnPlateau(
        optimizer,
        "max",
        factor=0.5,
        patience=2,
        verbose=True,
        threshold=1e-3,
        threshold_mode="abs",
    )

    aug = augment.Augment().cuda()

    if args.fp16:
        import apex.amp as amp

        net, optimizer = amp.initialize(net, optimizer, opt_level="O2")

    batch_iterator = iter(train_loader)
    sum_accuracy = 0
    step = 0
    config["eval_period"] = len(train_loader.dataset) // args.batch_size
    config["verbose_period"] = config["eval_period"] // 5

    train_start_time = time.time()
    for iteration in range(
        args.resume + 1,
        args.max_epoch * len(train_loader.dataset) // args.batch_size,
    ):
        t0 = time.time()
        try:
            sounds, type_ids = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(train_loader)
            sounds, type_ids = next(batch_iterator)
        except Exception as ex:
            print("Loading data exception:", ex)

        if torch.cuda.is_available():
            sounds = Variable(sounds.cuda())
            type_ids = Variable(type_ids.cuda())
        else:
            sounds = Variable(sounds)
            type_ids = Variable(type_ids)

        sounds = sounds.unsqueeze(3)
        sounds = sounds.permute(0, 3, 1, 2).float()

        if torch.cuda.is_available():
            one_hot = torch.cuda.FloatTensor(
                type_ids.shape[0], config["num_classes"]
            )
        else:
            one_hot = torch.FloatTensor(
                type_ids.shape[0], config["num_classes"]
            )
        one_hot.fill_(0.5 / (config["num_classes"] - 1))
        one_hot.scatter_(1, type_ids.unsqueeze(1), 0.5)

        # augmentation
        sounds = aug(sounds)
        # forward
        out = net(sounds)

        # backprop
        optimizer.zero_grad()
        loss = torch.sum(-one_hot * F.log_softmax(out, -1), -1).mean()
        # loss = F.cross_entropy(out, type_ids)

        if args.fp16:
            import apex.amp as amp

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        t1 = time.time()

        if iteration % config["verbose_period"] == 0:
            # accuracy
            _, predict = torch.max(out, 1)
            correct = (predict == type_ids)
            accuracy = correct.sum().item() / correct.size()[0]
            print(
                "iter: %d loss: %.4f | acc: %.4f | time: %.4f sec."
                % (iteration, loss.item(), accuracy, (t1 - t0)),
                flush=True,
            )
            sum_accuracy += accuracy
            step += 1

        warmup_steps = config["verbose_period"]
        if iteration < warmup_steps:
            warmup_learning_rate(optimizer, iteration, warmup_steps)

        if (
            iteration % config["eval_period"] == 0
            and iteration != 0
            and step != 0
        ):
            with torch.no_grad():
                loss, accuracy = evaluate(net, eval_loader)
            hours = int(time.time() - train_start_time) // 3600
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            print(
                "[{}] [{}] Eval accuracy:{:6f} | Train accuracy:{:6f}".format(
                    now, hours, accuracy, sum_accuracy / step
                ),
                flush=True,
            )
            scheduler.step(accuracy)
            sum_accuracy = 0
            step = 0

        if iteration % config["eval_period"] == 0 and iteration != 0:
            # save checkpoint
            print("Saving state, iter:", iteration, flush=True)
            save_ckpt(net, iteration)

    # final checkpoint
    save_ckpt(net, iteration)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epoch",
        default=100,
        type=int,
        help="Maximum epoches for training",
    )
    parser.add_argument(
        "--dataset_root",
        default="/media/data2/song/V5.npy",
        type=str,
        help="Root path of data",
    )
    parser.add_argument(
        "--lr", default=0.1, type=float, help="Initial learning rate"
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="Momentum value for optimizer",
    )
    parser.add_argument(
        "--resume",
        default=0,
        type=int,
        help="Checkpoint steps to resume training from",
    )
    parser.add_argument(
        "--finetune",
        default=False,
        type=bool,
        help="Finetune model by using all categories",
    )
    parser.add_argument(
        "--fp16",
        default=False,
        type=bool,
        help="Use float16 precision to train",
    )
    args = parser.parse_args()

    t0 = time.time()
    list_loader = ListLoader(
        args.dataset_root, config["num_classes"]
    )
    list_loader.export_labelmap()
    sound_list, train_indices, eval_indices = list_loader.sound_indices()

    train_set = BirdsDataset(sound_list, train_indices, True, args.finetune)
    eval_set = BirdsDataset(sound_list, eval_indices, False)
    print("train set: {} eval set: {}".format(len(train_set), len(eval_set)))
    eval_set.export_samples()

    train_loader = data.DataLoader(
        train_set,
        args.batch_size,
        num_workers=config["num_workers"],
        worker_init_fn=BirdsDataset.worker_init_fn,
        shuffle=True,
        pin_memory=True,
        collate_fn=BirdsDataset.my_collate,
    )
    eval_loader = data.DataLoader(
        eval_set,
        args.batch_size,
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=True,
        collate_fn=BirdsDataset.my_collate,
    )
    t1 = time.time()
    print("Load dataset with {} secs".format(t1 - t0))

    train(args, train_loader, eval_loader)
