import os
import time
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from shutil import copyfile
from tqdm import tqdm
from torch.utils import data
from torch.optim.adadelta import Adadelta
from sklearn.model_selection import train_test_split



from Visualizer import plot_graphs, plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from Network.stgcn_2stream import *
import csv
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, parameter_count
from thop import profile



device = 'cuda'
epochs = 30
batch_size = 32

# DATA FILES.
# Should be in format of
#  inputs: (N_samples, time_steps, graph_node, channels),
#  labels: (N_samples, num_class)
#   and do some of normalizations on it. Default data create from:
#       Data.create_dataset_(1-3).py
# where
#   time_steps: Number of frame input sequence, Default: 30
#   graph_node: Number of node in skeleton, Default: 14
#   channels: Inputs data (x, y and scores), Default: 3
#   num_class: Number of pose class to train, Default: 7
model_name = 'STGCN_2S'
dataset_name = "URFD_3classes"
save_folder = f'Result/{dataset_name}/{model_name}_{time.strftime("%Y%m%d%H%M%S")}'
train_data_file = f'DataFiles/{dataset_name}/train.pkl'
val_data_file = f'DataFiles/{dataset_name}/val.pkl'
test_data_file = f'DataFiles/{dataset_name}/test.pkl'
eval_only = False
# class_names = ['Not fall', 'Fall']
class_names = ['Not fall', 'Falling', 'Fall']

num_class = len(class_names)


def load_dataset(data_files, batch_size, split_size=0):
    """Load data files into torch DataLoader with/without spliting train-test.
    """
    features, labels = [], []
    for fil in data_files:
        with open(fil, 'rb') as f:
            fts, lbs = pickle.load(f)
            features.append(fts)
            labels.append(lbs)
        del fts, lbs
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    if split_size > 0:
        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=split_size,
                                                              random_state=9)
        train_set = data.TensorDataset(torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_train, dtype=torch.float32))
        valid_set = data.TensorDataset(torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_valid, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_set, batch_size)
    else:
        train_set = data.TensorDataset(torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(labels, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = None
    return train_loader, valid_loader


def accuracy_batch(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).mean()


def set_training(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)
    return model

def fnp(model, input=None):
    params = parameter_count(model)['']
    param_str = 'Parameter Size: {:.8f} M'.format(params / 1024 / 1024)
    if input is not None:
        flops = FlopCountAnalysis(model, input).total()
        flop_str = 'FLOPs: {:.8f} G'.format(flops / 1024 / 1024 / 1024)
        return param_str + "\n" + flop_str
    return param_str

if __name__ == '__main__':
    save_folder = os.path.join(os.path.dirname(__file__), save_folder)
    if not os.path.exists(save_folder):

        os.makedirs(save_folder)

    # DATA.
    # train_loader, _ = load_dataset(data_files[0:1], batch_size)
    # valid_loader, train_loader_ = load_dataset(data_files[1:2], batch_size, 0.2)
    train_loader, _ = load_dataset([train_data_file], batch_size)
    valid_loader, _ = load_dataset([val_data_file], batch_size)
    test_loader, _ = load_dataset([test_data_file], batch_size)
    # train_loader = data.DataLoader(data.ConcatDataset([train_loader.dataset, train_loader_.dataset]),
    #                                batch_size, shuffle=True)
    dataloader = {'train': train_loader, 'valid': valid_loader}
    # del train_loader_

    # MODEL.
    graph_args = {'strategy': 'spatial'}
    if (model_name == 'STGCN_1S'):
        model =OneStream_STGCN(graph_args=graph_args, num_class=num_class).to(device)
    elif (model_name == "STGCN_2S"):
        model = TwoStream_STGCN(graph_args=graph_args, num_class=num_class).to(device)
    
    # Use torchinfo to summarize the model
    input_shape = tuple(train_loader.dataset[0][0].shape)
        
    # Create a fake input from input shape
    fake_input = torch.zeros((batch_size,) + input_shape).to(device)

    # Print the output shape
    param_flop_str = fnp(model, fake_input)


    # Save model information (summary) in "model_info.txt"
    with open(os.path.join(save_folder, 'model_info.txt'), 'w', encoding='utf-8') as f:
        f.write(str(summary(model, input_size=(batch_size,) + input_shape)))
        f.write("\n")
        print(param_flop_str)
        f.write(param_flop_str)

        macs, params = profile(model, inputs=(fake_input, ))
        print("\nParams(M): %.7f \nFLOPs(G)\n: %.7f" % (params / (1000 ** 2), macs / (1000 ** 3)))
        f.write("\nParams(M): %.7f \nFLOPs(G): %.7f" % (params / (1000 ** 2), macs / (1000 ** 3)))
                
    losser = torch.nn.BCELoss()

    if (eval_only == False):
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer = Adadelta(model.parameters())


        # TRAINING.
        loss_list = {'train': [], 'valid': []}
        accu_list = {'train': [], 'valid': []}
        for e in range(epochs):
            print('Epoch {}/{}'.format(e, epochs - 1))
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model = set_training(model, True)
                else:
                    model = set_training(model, False)

                run_loss = 0.0
                run_accu = 0.0
                with tqdm(dataloader[phase], desc=phase) as iterator:
                    for pts, lbs in iterator:
                        # print(pts[:, :2, 1:, :])
                        # Create motion input by distance of points (x, y) of the same node
                        # in two frames.
                        pts = pts.to(device)
                        lbs = lbs.to(device)

                        # Forward.
                        out = model(pts)
                        loss = losser(out, lbs)
                        if phase == 'train':
                            # Backward.
                            model.zero_grad()
                            loss.backward()
                            optimizer.step()

                        run_loss += loss.item()
                        accu = accuracy_batch(out.detach().cpu().numpy(),
                                            lbs.detach().cpu().numpy())
                        run_accu += accu

                        iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
                            loss.item(), accu))
                        iterator.update()
                        #break
                loss_list[phase].append(run_loss / len(iterator))
                accu_list[phase].append(run_accu / len(iterator))
                #break

            print('Summary epoch:\n - Train loss: {:.4f}, accu: {:.4f}\n - Valid loss:'
                ' {:.4f}, accu: {:.4f}'.format(loss_list['train'][-1], accu_list['train'][-1],
                                                loss_list['valid'][-1], accu_list['valid'][-1]))

            # SAVE.
            torch.save(model.state_dict(), os.path.join(save_folder, f'{model_name}.pth'))

            plot_graphs(list(loss_list.values()), list(loss_list.keys()),
                        'Last Train: {:.2f}, Valid: {:.2f}'.format(
                            loss_list['train'][-1], loss_list['valid'][-1]
                        ), 'Loss', xlim=[0, epochs],
                        save=os.path.join(save_folder, 'loss_graph.png'))
            plot_graphs(list(accu_list.values()), list(accu_list.keys()),
                        'Last Train: {:.2f}, Valid: {:.2f}'.format(
                            accu_list['train'][-1], accu_list['valid'][-1]
                        ), 'Accu', xlim=[0, epochs],
                        save=os.path.join(save_folder, 'accu_graph.png'))
            
        # Save loss_list and accu_list to a CSV file
        with open(os.path.join(save_folder, 'log.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Valid Loss', 'Valid Accuracy'])
            for epoch in range(epochs):
                writer.writerow([epoch, loss_list['train'][epoch], accu_list['train'][epoch], loss_list['valid'][epoch], accu_list['valid'][epoch]])
            #break

        del train_loader, valid_loader

    
    model.load_state_dict(torch.load(os.path.join(save_folder, f'{model_name}.pth')))

    # EVALUATION.
    model = set_training(model, False)

    eval_loader, _ = load_dataset([test_data_file], 32)

    print('Evaluation.')
    run_loss = 0.0
    run_accu = 0.0
    y_preds = []
    y_trues = []
    total_time = 0.0
    num_samples = 0

    with tqdm(eval_loader, desc='eval') as iterator:
        for pts, lbs in iterator:
            pts = pts.to(device)
            lbs = lbs.to(device)
            start_time = time.time()

            out = model(pts)
            end_time = time.time()

            loss = losser(out, lbs)

            run_loss += loss.item()
            accu = accuracy_batch(out.detach().cpu().numpy(),
                                  lbs.detach().cpu().numpy())
            run_accu += accu

            y_preds.extend(out.argmax(1).detach().cpu().numpy())
            y_trues.extend(lbs.argmax(1).cpu().numpy())

            iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
                loss.item(), accu))
            iterator.update()

            total_time += end_time - start_time
            num_samples += pts.size(0)

    average_inference_time = total_time / num_samples
    run_loss = run_loss / len(iterator)
    run_accu = run_accu / len(iterator)

    plot_confusion_matrix(y_trues, y_preds, class_names, 'Eval on: {}\nLoss: {:.4f}, Accu{:.4f}'.format(
        os.path.basename(test_data_file), run_loss, run_accu
    ), 'true', save=os.path.join(save_folder, '{}-confusion_matrix.png'.format(
        os.path.basename(test_data_file).split('.')[0])))

    print('Eval Loss: {:.7f}, Accu: {:.7f}'.format(run_loss, run_accu))
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_trues, y_preds, average='weighted')
    recall = recall_score(y_trues, y_preds, average='weighted')
    f1 = f1_score(y_trues, y_preds, average='weighted')

    # Print the results
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)
    print("Average Inference Time: {:.7f} seconds".format(average_inference_time))

    # Save results to "result.txt" file
    with open(os.path.join(save_folder, 'result.txt'), "w") as f:
        f.write("Eval Loss: {:.7f}, Accu: {:.7f}\n".format(run_loss, run_accu))
        f.write("Precision: {:.7f}\n".format(precision))
        f.write("Recall: {:.7f}\n".format(recall))
        f.write("F1-score: {:.7f}\n".format(f1))
        f.write("Average Inference Time: {:.7f} seconds".format(average_inference_time))