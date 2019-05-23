import numpy as np
from matplotlib import pyplot as plt

d_loss = np.loadtxt("d_loss.txt")
g_loss = np.loadtxt("g_loss.txt")
q_loss = np.loadtxt("q_loss.txt")

plt.plot(np.arange(len(d_loss)), d_loss, label="d_loss")
plt.plot(np.arange(len(d_loss)), g_loss, label="g_loss")
plt.plot(np.arange(len(d_loss)), q_loss, label="q_loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig("loss.jpg")
plt.show()

probs_fake_after = np.loadtxt("probs_fake_after.txt")
probs_fake_before = np.loadtxt("probs_fake_before.txt")
probs_real = np.loadtxt("probs_real.txt")

plt.plot(np.arange(len(d_loss)), probs_fake_after, label="prob. of fake after update")
plt.plot(np.arange(len(d_loss)), probs_fake_before, label="prob. of fake before update")
plt.plot(np.arange(len(d_loss)), probs_real, label="prob. of real")
plt.title("Probability Curve")
plt.legend()
plt.savefig("prob.jpg")
plt.show()
