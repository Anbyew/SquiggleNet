import os
import click
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Parameters
params = {'batch_size': 1000,
          'shuffle': True,
          'num_workers': 10}

max_epochs = 20


zm_train = "zymo_train_totmad_41_400000.pt"
hl_train = "hela_train_totmad_41_400000.pt"
zm_val = "zymo_val_totmad_4_20000.pt"
hl_val = "hela_val_totmad_4_20000.pt"

training_set = Dataset(zm_train, hl_train)
training_generator = DataLoader(training_set, **params1)

validation_set = Dataset(zm_val, hl_val)
validation_generator = DataLoader(validation_set, **params1)

zymo_train = torch.load(os.path.join(data_path, zm_train))
hela_train = torch.load(os.path.join(data_path, hl_train))

zymo_val = torch.load(os.path.join(data_path, zm_val))
hela_val = torch.load(os.path.join(data_path, hl_val))


def solver(learning_rate=1e-3):
  model = ResNet(Bottleneck, [2,2,2,2]).to(device)
  # tpp = torch.load(os.path.join(model_path, "model_crona_1.ckpt"))
  # model.load_state_dict(tpp)

  criterion = nn.CrossEntropyLoss().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  bestacc = 0
  bestmd = None

  i = 0

  for epoch in range(max_epochs):
    # Training
    for spx, spy in training_generator:
        spx, spy = spx.to(device), spy.to(torch.long).to(device)

        # Forward pass
        outputs = model(spx)
        loss = criterion(outputs, spy)
        acc = 100.0 * (spy == outputs.max(dim=1).indices).float().mean().item()

        # Validation
        with torch.set_grad_enabled(False):
            acc_vt = 0
            vti = 0
            for valx, valy in validation_generator:
                valx, valy = valx.to(device), valy.to(device)
                outputs_val = model(valx)
                acc_v = 100.0 * (valy == outputs_val.max(dim=1).indices).float().mean().item()
                vti += 1
                acc_vt += acc_v
            acc_vt = acc_vt / vti
            if bestacc < acc_vt:
                bestacc = acc_vt
                bestmd = model
                torch.save(bestmd.state_dict(), os.path.join(model_path, "model_crona_v2_1.ckpt"))
        
            print("epoch: " + str(epoch) + ", i: " + str(i) + ", bestacc: " + str(bestacc) + ", curacc: " + str(acc_vt) + \
                  ", trainacc: " + str(acc) + ", loss: " + str(loss.cpu()))
            i += 1
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  return bestacc, bestmd

@click.command()
@click.option('--tTrain', '-tt', help='The path of target sequence training set', type=click.Path(exists=True))
@click.option('--tVal', '-tv', help='The path of target sequence validation set', type=click.Path(exists=True))
@click.option('--nTrain', '-nt', help='The path of non-target sequence training set', type=click.Path(exists=True))
@click.option('--nVal', '-nv', help='The path of non-target sequence validation set', type=click.Path(exists=True))
@click.option('--outpath', '-o', help='The output path and name for the best trained model')
@click.option('--interm', '-i', help='The path and name for model checkpoint (optional)', type=click.Path(exists=True))


def main(tTrain, tVal, nTrain, nVal, outpath, interm):

if __name__ == '__main__':
  main()
