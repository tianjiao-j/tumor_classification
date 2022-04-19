import matplotlib.pyplot as plt

# read data
file = 'resnet50/log.txt'
epochs = []
train_loss = []
train_acc = []
val_acc = []
val_loss = []
with open(file, 'r') as f:
     for line in f.readlines():
         line_list = line.strip().split()
         epochs.append(int(line_list[0].split(":")[-1]))
         train_loss.append(float(line_list[2].split(":")[-1]))
         train_acc.append(float(line_list[4].split(":")[-1]))
         val_loss.append(float(line_list[6].split(":")[-1]))
         val_acc.append(float(line_list[-1].split(":")[-1]))

# plot figures
plt.title("ResNet-50 Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(epochs, train_loss, label= 'train_loss')
plt.plot(epochs, val_loss, label= 'val_loss')
plt.legend()
plt.savefig("figures/50loss.png")
plt.show()

plt.title("ResNet-50 Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(epochs, train_acc, label= 'train_accuracy')
plt.plot(epochs, val_acc, label = 'val_accuracy')
plt.legend()
plt.savefig("figures/50acc.png")
plt.show()


