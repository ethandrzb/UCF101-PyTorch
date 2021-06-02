import timeit
import os
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from Utils import build_paths
import C3D_Borrowed
from Dataset import UCF101


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\nDevice being used:', device)


nEpochs = 100           # Number of epochs for training
resume_epoch = 0        # Default is 0, change if want to resume
useTest = True          # See evolution of the test set when training
nTestInterval = 5       # Run on test set every nTestInterval epochs
snapshot = 50           # Store a model every snapshot epochs
lr = .00001             # Learning rate
num_classes = 10
batch_size = 20
print_batch = True
pretrain = False

print('Batch Size: %d' % batch_size)
print('Pretrain: %s' % pretrain)
print('Learning Rate: %f' % lr)

class_idx, train_split, test_split, frames_root, pret_path = build_paths()





def train_model(num_classes=num_classes, lr=lr, num_epochs=nEpochs, save_epoch=snapshot,
                useTest=useTest, test_interval=nTestInterval):


    model = C3D_Borrowed.C3D(num_classes=num_classes, pret_path=pret_path, pretrained=pretrain)
    # print(model.parameters())
    # sys.exit()

    ### ▼ ############################################### ▼ ###################################################### ▼ ###
    train_params = [{'params': C3D_Borrowed.get_1x_lr_params(model), 'initial_lr': lr},
                    {'params': C3D_Borrowed.get_10x_lr_params(model), 'initial_lr': lr * 10}]
    ### ▲ ############################################### ▲ ###################################################### ▲ ###

    criterion = nn.CrossEntropyLoss()

    ### ▼ ############################################### ▼ ###################################################### ▼ ###
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    # train_params vs. model.parameters

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the initial_lr by 10 every 10 epochs

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    ### ▲ ############################################### ▲ ###################################################### ▲ ###

    model.to(device)

    ### ▼ ############################################### ▼ ###################################################### ▼ ###
    criterion.to(device)
    ### ▲ ############################################### ▲ ###################################################### ▲ ###

    trainset = UCF10(class_idxs=class_idx, split=train_split, frames_root=frames_root,
                     clip_len=16)

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = UCF10(class_idxs=class_idx, split=test_split, frames_root=frames_root,
                    clip_len=16)

    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)


    trainset_size = len(trainset)
    trainloader_size = len(train_dataloader)
    testset_size = len(testset)
    testloader_size = len(test_dataloader)

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step

        start_time = timeit.default_timer()

        # reset the running loss and corrects
        running_loss = 0.0
        running_corrects = 0.0

        scheduler.step()
        model.train()
        batch_idx = 0
        for inputs, labels in tqdm(train_dataloader):
            # move inputs and labels to the device the training is taking place on
            inputs = Variable(inputs, requires_grad=True).to(device)
            labels = Variable(labels).to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            batch_idx += 1
            total = batch_idx*batch_size
            if print_batch:
                print('\nEpoch: %d | Batch: %d/%d | Loss: %.3f | Acc: %.3f%% [Train]'
                      % (epoch, batch_idx, trainloader_size, running_loss/total, 100.*running_corrects/total))

        epoch_loss = running_loss / trainset_size
        epoch_acc = running_corrects.double() / trainset_size

        print("[Train] Epoch: {}/{} | initial_lr: {} |  Loss: {} | Acc: {:.3f}".format(epoch+1, nEpochs, lr, epoch_loss, 100*epoch_acc.item()))
        stop_time = timeit.default_timer()
        print('Execution time: %.2f min \n' % ((stop_time - start_time)/60))


        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / testset_size
            epoch_acc = running_corrects.double() / testset_size

            print("[Test] Epoch: {}/{} | Loss: {} | Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print('Execution time: %.2f min \n' % ((stop_time - start_time) / 60))

        if (epoch + 1) % 10 == 0:
            print('Saving Checkpoint..')
            state = {'net': model.state_dict(), 'acc': epoch_acc, 'epoch': (epoch+1)}
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')


if __name__ == "__main__":
    train_model()

# [5, 5, 8, 8, 1, 5, 5, 6, 6, 4, 1, 8, 0, 9, 7, 7, 4, 7, 0, 4
# [7, 5, 9, 9, 5, 0, 8, 4, 2, 0, 5, 6, 2, 1, 0, 0, 4, 5, 0, 6

# [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0
# [5, 1, 0, 5, 4, 6, 3, 7, 2, 8, 5, 3, 7, 4, 3, 2, 0, 7, 9, 1])