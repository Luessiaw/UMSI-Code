# 可视化
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as matcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MultipleLocator
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import colors

import numpy as np
from .mathTools import *

plt.rcParams['text.usetex'] = True # 支持 latex
plt.rcParams['font.family'] = ["Times New Roman"]
# plt.rcParams['text.usetex'] = True
# 使用 xelatex 支持中文
# plt.rcParams.update({
#     "text.usetex": True,
#     "pgf.texsystem": "xelatex",  # 指定使用 xelatex
#     "pgf.preamble": r"""
#         \usepackage{xeCJK}
#         \setCJKmainfont{SimHei}  % 替换为系统中的中文字体，例如 SimSun, SimKai
#     """
# })


def format_func(x, pos):
    '''使刻度值显示为 100 倍'''
    return '{:.0f}'.format(x * 100)

def get3dAx(fig=None,axisNum=-1):
    if not fig:
        fig = plt.figure()
    axes = fig.axes
    if len(axes) < 1:
        ax = fig.add_subplot(projection="3d")
    else:
        ax = fig.axes[axisNum]
    return fig,ax

def get2dAx(fig=None,axisNum=-1):
    if not fig:
        fig = plt.figure()
    axes = fig.axes
    if len(axes) < 1:
        ax = fig.add_subplot()
    else:
        ax = fig.axes[axisNum]
    return fig,ax

def getDimAx(fig=None,dim=3,axisNum=-1):
    if dim==3:
        fig,ax = get3dAx(fig,axisNum)
    else:
        fig,ax = get2dAx(fig,axisNum)
    return fig,ax

def setAxLabel(ax,unit="cm",x_sticks=[],y_sticks=[],z_sticks=[]):
    ax.set_xlabel(f'x ({unit})')
    ax.set_ylabel(f'y ({unit})')
    ax.set_zlabel(f'z ({unit})')

    ax.set_xticks(x_sticks)
    ax.set_yticks(y_sticks)
    ax.set_zticks(z_sticks)
    
    # ax.grid(True)
    ax.set_axis_off()





def setAxOn(fig=None):
    fig,ax = get3dAx(fig)
    ax.set_axis_on()
    return fig

def setAxOff(fig=None):
    fig,ax = get3dAx(fig)
    ax.set_axis_off()
    return fig

def show(*args, **kwargs):
    return plt.show(*args, **kwargs)

def draw(*args, **kwargs):
    return plt.draw(*args, **kwargs)

def plot2dPoint(point,fig=None,ax=None,*args, **kwargs):
    if not ax:
        fig,ax = get2dAx(fig)
    
    ax.plot(*point,*args,**kwargs)
    return fig


def plot2dPoints(points:list,fig=None,ax=None,returnsc=False,*args, **kwargs):
    if not ax:
        fig,ax = get2dAx(fig)
    if ax and not fig:
        fig = ax.figure
    xs = []
    ys = []
    for point in points:
        xs.append(point[0])
        ys.append(point[1])
    
    sc = ax.scatter(xs,ys,*args,**kwargs)
    if returnsc:
        return fig,sc
    else:
        return fig

def plot2dCircle(center,radius,fig=None,ax=None,*args, **kwargs):
    if not ax:
        fig,ax = get2dAx(fig)
    
    circle = Circle(center, radius, *args, **kwargs)
    ax.add_patch(circle)
    return fig

def plot3dPoints(points:list,fig=None,ax=None,*args,**kwargs):
    if not ax:
        fig,ax = get3dAx(fig)
    
    xs = []
    ys = []
    zs = []
    for point in points:
        xs.append(point[0])
        ys.append(point[1])
        zs.append(point[2])
    
    ax.scatter(xs,ys,zs,*args,**kwargs)
    return fig

def plot3dPoint(point,fig=None,*args, **kwargs):
    return plot3dPoints([point],fig,*args, **kwargs)

def plot3dScatter(points,values,fig=None,ulabel="Amplitude",*args,**kwargs):
    fig,ax = get3dAx(fig)
    
    xs = []
    ys = []
    zs = []
    for point in points:
        xs.append(point[0])
        ys.append(point[1])
        zs.append(point[2])
    
    scatter = ax.scatter(xs,ys,zs,c=values, marker='o',*args,**kwargs)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(ulabel, rotation=270, labelpad=15)
    return fig

def plot2dScatter(xs,ys,values,fig=None,ax=None,*args,**kwargs):
    if not ax:
        fig,ax = get3dAx(fig)

    ignorecbar = kwargs.pop("ignorecbar",False)
    scatter = ax.scatter(xs,ys,c=values,*args,**kwargs)
    
    if not ignorecbar:
        cbar = fig.colorbar(scatter)
        cbar.ax.tick_params(labelsize=20)
    return fig

def plot3dHarfSphere(center,radius,n,fig=None,*args, **kwargs):
    '''以 center 为球心，radius 为半径, n 的方向为对称轴画一个半球面'''
    R = rotationMatrixFromNToM(unit_z,n)

    u,v = np.mgrid[0:2*np.pi:50j,0:np.pi/2:50j]
    X1 = np.cos(u)*np.sin(v)
    Y1 = np.sin(u)*np.sin(v)
    Z1 = np.cos(v)
    
    points = np.stack((X1, Y1, Z1))*radius
    # points = np.dot(R, points)
    points = np.einsum("ij,jkl -> ikl",R,points)
    X,Y,Z = points
    X += center[0]
    Y += center[1]
    Z += center[2]

    fig,ax = get3dAx(fig)
    ax.plot_surface(X,Y,Z,*args, **kwargs)
    # setAxLabel(ax)
    return fig

def plot3dSphere(center,radius,fig=None,*args, **kwargs):
    '''以 center 为球心，radius 为半径的球面'''

    u,v = np.mgrid[0:2*np.pi:50j,0:np.pi:50j]
    X = np.cos(u)*np.sin(v)*radius
    Y = np.sin(u)*np.sin(v)*radius
    Z = np.cos(v)*radius
    
    X += center[0]
    Y += center[1]
    Z += center[2]

    fig,ax = get3dAx(fig)
    ax.plot_surface(X,Y,Z,*args, **kwargs)
    # setAxLabel(ax)
    return fig

def plotSphere(center,radius,ax:plt.Axes,*args, **kwargs):
    '''以 center 为球心，radius 为半径的球面'''

    u,v = np.mgrid[0:2*np.pi:50j,0:np.pi:50j]
    X = np.cos(u)*np.sin(v)*radius
    Y = np.sin(u)*np.sin(v)*radius
    Z = np.cos(v)*radius
    
    X += center[0]
    Y += center[1]
    Z += center[2]

    ax.plot_surface(X,Y,Z,*args, **kwargs)


def plot3dCylinder(radius,z_range,fig=None,*args,**kwargs):
    # 生成圆柱的点坐标
    z1,z2 = z_range
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, Z = np.meshgrid(theta, np.linspace(z1, z2, 50))
    X = radius * np.cos(theta_grid)
    Y = radius * np.sin(theta_grid)
    
    fig,ax = get3dAx(fig)
    ax.plot_surface(X,Y,Z,*args, **kwargs)
    # setAxLabel(ax)
    return fig

def plot3dBox(x_range,y_range,z_range,fig=None,*args,**kwargs):
    x1,x2 = x_range
    y1,y2 = y_range
    z1,z2 = z_range
    # 扩展Z值为二维数组
    Z1 = np.full((2, 2), z1)  # 底面
    Z2 = np.full((2, 2), z2)  # 顶面   
    Z3 = np.array([[z1,z2],[z1,z2]]) 
    
    fig,ax = get3dAx(fig)
    # 绘制盒子的六个面
    # 底面
    ax.plot_surface([[x1,x1],[x2,x2]], [[y1,y2],[y1,y2]], Z1, *args, **kwargs)
    # 顶面
    ax.plot_surface([[x1,x1],[x2,x2]], [[y1,y2],[y1,y2]], Z2, *args, **kwargs)
    # 前面
    ax.plot_surface([[x1,x1],[x1,x1]], [[y1,y2],[y1,y2]], Z3, *args, **kwargs)
    # 后面
    ax.plot_surface([[x2,x2],[x2,x2]], [[y1,y2],[y1,y2]], Z3,*args, **kwargs)
    # 左侧面
    ax.plot_surface([[x1,x1],[x1,x1]], [[y1,y1],[y2,y2]], Z3, *args, **kwargs)
    # 右侧面
    ax.plot_surface([[x2,x2],[x2,x2]], [[y1,y1],[y2,y2]], Z3, *args, **kwargs)

    return fig

def plot3dArrow(point1,n,fig=None,*args, **kwargs):
    fig,ax = get3dAx(fig)
    x,y,z = point1
    u,v,w = n
    ax.quiver(x,y,z,u,v,w,*args, **kwargs)
    return fig

def plot3dArrows(points,ns,fig=None,*args,**kwargs):
    '''绘制许多箭头'''
    fig,ax = get3dAx(fig)
    for i in range(len(ns)):
        point = points[i]
        n = ns[i]
        fig = plot3dArrow(point,n,fig,*args, **kwargs)
    return fig

def plot3dSegment(point1,point2,fig=None,*args, **kwargs):
    fig,ax = get3dAx(fig)
    x = [point1[0],point2[0]]
    y = [point1[1],point2[1]]
    z = [point1[2],point2[2]]
    ax.plot(x,y,z,*args, **kwargs)
    return fig

def plot_surface(x,y,z,u,fig=None,*args, **kwargs):
    fig,ax = get3dAx(fig)

    # 设置图形标题和颜色映射
    color_dimension = u # change to desired fourth dimension
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matcolors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)    
    # plt.colorbar(surf, shrink=0.5, aspect=5)  # 添加颜色映射条
    ax.plot_surface(x,y,z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False,*args, **kwargs)
    return fig

def read_matrix_from_txt(filename:str):
    '''从文件中读取模拟结果'''
    # 读取文件
    with open(filename, 'r',encoding="utf-8") as file:
        lines = file.readlines()
    # 提取矩阵数据
    matrix_data = [line.strip().split(',') for line in lines[:3]]
    # 将数据转换为 NumPy 数组
    matrix_array = np.array(matrix_data, dtype=float)
    return matrix_array

def read_array_from_txt(filename:str,data0func=None):
    '''从文件中读取模拟结果，其中第一行是探头阵列的大小，为数组'''
    with open(filename,"r",encoding="utf-8") as file:
        lines = file.readlines()
    # data0 = lines[0].strip().split('), (')
    if data0func:
        data0 = data0func(lines[0])
    else:
        data0 = lines[0].strip().split(',')
    # data0 = [i**2 for i in range(2,10)]
    data = [line.strip().split(',') for line in lines[1:3]]
    data.insert(0,data0)
    data = np.array(data)
    return data


def format_ticks(x, pos):
    # 将 x 转换为 pi/2 的倍数
    n = int(np.round(4 * x / np.pi))  # 将 x / pi 调整为 pi/4 的倍数
    if n == 0:
        return '0'
    elif n == 1:
        return '${\\pi}/{4}$'
    elif n == 2:
        return '${\\pi}/{2}$'
    elif n == 3:
        return '${3\\pi}/{4}$'
    elif n == 4:
        return '$\\pi$'

def plotXYs(X,Ys,fig=None,*args,**kwargs):
    '''X为横轴，Ys为多个曲线'''
    fig = get2dAx(fig)
    for i in range(Ys.shape[0]):
        plt.plot(X,Ys[i,:],*args,**kwargs)
    return fig

def showGridPositionAndIndex(gridPoints:list[np.ndarray],fig=None,ax=None,dim=2,*args, **kwargs):
    '''画出格点与序号的关系。'''
    points = np.asarray(gridPoints)
    if dim==3:
        print("Not supported yet.")
    elif dim==2:
        xs = points[:,0]
        ys = points[:,2]
        values = np.arange(xs.size)
        fig = plot2dScatter(xs,ys,values=values,*args, **kwargs,fig=fig,ax=ax)
    return fig

