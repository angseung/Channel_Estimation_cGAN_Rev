import matplotlib.pyplot as plt
import numpy as np

nmse_3D = np.load("NMSE_3D.npy")[1:, :, :].mean(axis=0)

### For Comparison

fig_comp = plt.figure(figsize=(10, 5))
marker_list = [
    "-o",
    "-v",
    "-^",
    "-<",
    "->",
    "-1",
    "-2",
    "-3",
    "-4",
    "-p",
    "-P",
    "-+",
    "-x",
    "-d",
    "-8",
    "-*",
] * 2
nmse_ori = np.array(
    [-10.2, -13.1, -14.7, -15.9, -16.1, -16.2, -16.2, -16.21, -16.22, -16.2, -16.2]
)

for i in range(nmse_3D.shape[0]):
    plt.plot(
        range(-10, 41, 5),
        np.squeeze(nmse_3D[i, :]),
        marker_list[i],
        label="MPCs = %d" % (i + 3),
    )

aaa = np.load("NMSE_refer_val.npy")[2, :]
plt.plot(range(-10, 41, 5), nmse_ori, "k*--", label="reference")
plt.plot(range(-10, 41, 5), aaa, "bo--", label="(a)")
plt.legend(loc="best", ncol=5)
plt.ylim([-17, 0])
plt.grid(True)
plt.xlabel("SNR (dB)", fontsize="large")
plt.ylabel("NMSE (dB)", fontsize="large")
plt.title("NMSE comparison for all MPCs", fontsize="x-large")
plt.show()

### For validate performance

fig_val = plt.figure()

for i in range(nmse_3D.shape[1]):
    if i < 4:
        dB = 5 * i - 10
        labels = "SNR = %ddB" % dB
        plt.plot(range(3, 26), np.squeeze(nmse_3D[:, i]), marker_list[i], label=labels)

    elif i == 10:
        labels = "SNR = 10~40dB"
        plt.plot(range(3, 26), np.squeeze(nmse_3D[:, i]), marker_list[i], label=labels)


plt.grid(True)
plt.legend(loc="best")
plt.xlabel("The number of MPCs", fontsize="large")
plt.ylabel("NMSE (dB)", fontsize="large")
plt.title("NMSE validation for all SNR conditions", fontsize="x-large")
plt.show()
