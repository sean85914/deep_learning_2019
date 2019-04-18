'''
  Plot confusion matric
  Editor: Sean Lu
  Last Edited: 4/17
'''
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from dataset import RetinopathyLoader

model = torch.load("model/temp/resnet_50_epoch_10_batch_15_lr_0.001_pretrained_acc_0.822.pth")

transform_test  = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

device = 'cuda'
rpl_test  = RetinopathyLoader("data",  "test", transform_test)
num_workers = 4
batch_size = 8
test_loader = Data.DataLoader(rpl_test, batch_size, shuffle=False, num_workers=num_workers)

predicted_list = []
gt_list = []
for batch_idx, (data, target) in enumerate(test_loader):
    print("[{}/{}]...\r".format(batch_idx, len(test_loader)), end="")
    data = data.to(device)
    out = model(data)
    _, predicted = torch.max(out, 1)
    predicted_np = predicted.cpu().numpy().tolist()
    target_np = target.numpy().tolist()
    gt_list += target_np
    predicted_list += predicted_np
print("End prediction")

classes = ["0", "1", "2", "3", "4"]
cm = confusion_matrix(gt_list, predicted_list)
# Normalize
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
# We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=classes, yticklabels=classes,
       title="ResNet-50 w/ Pretrained Confusion Matrix",
       ylabel='True label',
       xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
fmt = '.2f'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

plt.savefig("confusion_matrix.png")
