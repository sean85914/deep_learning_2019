'''
  Train model function
  Editor: Sean Lu
  Last Edited: 4/17
'''
import time

def train_model(model, dataloader, net_type, have_pretrained, criterion, optimizer, idx):
    since = time.time()
    model.train() # Set model to training mode
    loss_now = None
    for batch_idx, (data, target) in enumerate(dataloader):
        print("[{}/{}]...\r".format(batch_idx, len(dataloader)), end="")
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        loss_now = loss.item()
    cost_time = time.time()-since
    print('Train epoch: {}\t | Loss: {:.6f} \t | Cost time: {:} m: {} s'.
         format(idx, loss_now, int(cost_time/60), int(cost_time%60)))
