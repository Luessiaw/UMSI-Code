# data from fig4c-data.py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
import glob

dataFolder = "figs-original/data"
saveFolder = "figs-original/subplots"

def getErr(filePath):
    data = np.load(filePath)
    err = data["errs"]
    err *= 1e2
    vmin = np.min(err)
    vmax = np.max(err)
    return err,vmin,vmax

def drawHarfCircle(ax:plt.Axes):
    # 生成半圆的参数点（上半圆）
    radius = 10
    theta = np.linspace(0, np.pi, 100)  # 角度从0到π
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ax.plot(x, y, 'k-', linewidth=3)  # 绘制半圆曲线

labels = []
errs = []
vmins = []
vmaxs = []

filepaths = glob.glob(os.path.join(dataFolder,"fig4c-*.npz"))

for filepath in filepaths:
    err,vmin,vmax = getErr(filepath)
    errs.append(err)
    vmins.append(vmin)
    vmaxs.append(vmax)
    label = os.path.splitext(filepath)[0]
    label = label.split("\\")[-1].replace("fig4c-","")
    labels.append(label)

vmin = np.min(np.array(vmins))
vmax = np.max(np.array(vmaxs))

x = np.linspace(-10,10,26)
z = np.linspace(0,10,12)
X,Z = np.meshgrid(x, z)

xi = np.linspace(x.min(), x.max(), 200)
zi = np.linspace(z.min(), z.max(), 100)
Xi, Zi = np.meshgrid(xi, zi)

Ri = np.sqrt(Xi**2 + Zi**2)
mask = Ri>10
for i in range(len(labels)):
    label = labels[i]
    err = errs[i]
    erri = griddata((X.flatten(), Z.flatten()), err.flatten(), (Xi, Zi), method='cubic')
    erri[mask] = np.nan
    # erri = (erri - vmin)/(vmax-vmin)
    # erri = np.log10(erri)
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)
    im = ax.pcolormesh(Xi,Zi,erri,shading="auto",vmin=vmin,vmax=vmax,cmap="coolwarm")
    drawHarfCircle(ax)
    ax.set_aspect(1)
    ax.set_title(label)
    ax.set_xlabel("x (cm)")   
    ax.set_ylabel("z (cm)")
    ax.set_xlim([-10.5,10.5])
    ax.set_ylim([-0.5,10.5])
    
    # 获取主图的坐标范围
    ax_pos = ax.get_position()
    cbar_height = ax_pos.height  # 使用主图高度
    # 创建与主图等高的colorbar
    cax = fig.add_axes([ax_pos.x1 + 0.02, ax_pos.y0, 0.02, cbar_height])
    cb = plt.colorbar(im, cax=cax,label="Localization Accuracy (cm)")

    fig.savefig(os.path.join(saveFolder,f"fig4c-{label}.png"))
    plt.close(fig)

