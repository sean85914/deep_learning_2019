'''
  Test with given model and data loader and return accuracy
  Editor: Sean Lu
  Last Edited: 4/17
'''
import time
import torch

device = "cuda"

def test_model(model, dataloader, data_len, phase):
    since = time.time()
    #model.eval()
    correct_num = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        print("[{}/{}]...\r".format(batch_idx, len(dataloader)), end="")
        data, target = data.to(device), target.to(device)
        out = model(data)
        _, predicted = torch.max(out, 1)
        for i in range(len(target)):
            if predicted.cpu()[i] == target.cpu()[i]:
                correct_num += 1
    acc = correct_num/float(data_len)
    cost_time = time.time() - since
    print("Test against {} dataset, accuracy: {:.3f}, cost time: {} m: {} s".format
          (phase, acc, int(cost_time/60), int(cost_time%60)))
    return acc
