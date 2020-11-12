import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

import os
from sklearn.metrics import roc_auc_score


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
training_generator = DataLoader(training_set, **params)

validation_set = Dataset(zm_val, hl_val)
validation_generator = DataLoader(validation_set, **params)

zymo_val = torch.load(os.path.join(data_path, zm_val))
hela_val = torch.load(os.path.join(data_path, hl_val))

data_val = torch.cat((zymo_val, hela_val))
print("Total validation data shape: " + str(data_val.shape))
y_val = torch.cat((torch.zeros(zymo_val.shape[0]), torch.ones(hela_val.shape[0]))) #human: 1, others: 0
print("Total validation label shape: " + str(y_val.shape))


def solver(learning_rate=1e-3):
  model = ResNet(Bottleneck, [2,2,2,2]).to(device)
  # tpp = torch.load(os.path.join(model_path, "model_all_991_tmp_999_11.ckpt"))
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

        with torch.no_grad():
          valx = data_val.to(device)
          valy = y_val.to(torch.long).to(device)
          outputs_val = model(valx)
          acc_v = 100.0 * (valy == outputs_val.max(dim=1).indices).float().mean().item()
          auroc = roc_auc_score(valy.cpu(), outputs_val.max(dim=1).indices.cpu())
          if bestacc < auroc:
              bestacc = auroc
              bestmd = model
              torch.save(bestmd.state_dict(), os.path.join(model_path, "model_all_991_tmp_999_11.ckpt"))
          i += 1
          if i%100 == 0:
            print("epoch: " + str(epoch) + ", i: " + str(i) + ", bestauroc: " + str(bestacc) + ", curauroc: " + str(auroc) + ", acc: " + str(acc_v))

            i += 1
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  return bestacc, bestmd



bestacc, bestmd = solver()