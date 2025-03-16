# -*- coding: utf-8 -*-
"""
python 密度散点图各要素绘制，及统计占比
@e-mail:chinesevoice@163.com
任何问题可联系以上邮箱。
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def get_fit_para(test_target,pred_data):
    #计算拟合线坡度、截距、p值、RMSE、RPD等参数
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    std_measure = np.std(test_target,ddof=1)
    error = test_target - pred_data
    error_2 = error**2
    error_sum = error_2.sum()
    rmse = (error_sum/len(test_target))**0.5
    RPD = std_measure / rmse
    r_squared = r_value**2
    print('Linear regression slope: {:.2}'.format(slope))
    print('Linear regression intercept: {:.2}'.format(intercept))
    print('Linear regression r2: {:.2}'.format(r_squared))
    print('Linear regression p_value: {:.2}'.format(p_value))
    print('Linear regression RMSE: {:.2}'.format(rmse))
    print('Linear regression RPD: {:.2}'.format(RPD))
    return None

def count_nums(z_prob,level):
    print('特定等高线内数据比例:')
    for z_value in level:
        part_ = np.sum(z>z_value) / len(z)
        print('判定基准：',z_value)
        print('数据占比：{:.2%}'.format(part_))
    return None

#========第一步：产生随机正态二维数据，实际使用替换x，y即可=======================
# Define the range of values for x and y
x_min, x_max = 0, 0.3
y_min, y_max = 0, 0.3
# Define the mean and covariance matrix for the normal distribution
mean = [0.2, 0.2]  # Mean of zero for both x and y
cov = [[1, 0.5], [1, 1]]  # Identity covariance matrix
# Generate random samples from the normal distribution within the specified range
num_samples = 2000  # Number of data points to generate
samples = np.random.multivariate_normal(mean, cov, num_samples)
# Extract the x and y coordinates from the generated samples
x = samples[:, 0]
y = samples[:, 1]
# Scale the x and y coordinates to fit within the specified range
x = x_min + (x - np.min(x)) * (x_max - x_min) / (np.max(x) - np.min(x))
y = y_min + (y - np.min(y)) * (y_max - y_min) / (np.max(y) - np.min(y))

#=======================第二步：计算拟合线参数和概率密度=======================
x,y = np.array(x),np.array(y)
#1.拟合线参数打印
_ = get_fit_para(x,y)

#2.概率密度
xy = np.vstack([x, y])
# m = stats.gaussian_kde(xy)
# m.set_bandwidth(0.1)
# n = m.evaluate(xy)
z = stats.gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

#x此处为从小到大排序，此处参数与get_fit_para部分重合
a, b = np.polyfit(x, y, deg=1)
y_est = a * x + b
#=======================第三步：绘图=======================
# 绘图，可自行调整颜色等等
fig,ax=plt.subplots(figsize=(10,9))
scatter=ax.scatter(x, y, marker='o', c=z, edgecolors=None ,s=15,cmap='Spectral_r')
cbar=plt.colorbar(scatter,shrink=1,orientation='vertical',extend='both',pad=0.015,aspect=30)
cbar.ax.tick_params(labelsize = 22)
ax.plot([0,1],[0,1],'black',lw=1)  # 画的1:1线，线的颜色为black，线宽为0.8
ax.plot(x, y_est,'red',lw=1) #绘制拟合线
level = np.array([20,40,60,80]) #设置等高线特定值绘制
ax.tricontour(x, y, z,levels = level,colors='black',linewidths=1.0)
#计算特定等高线内数据比例
_ = count_nums(z,level)

plt.xlabel('Simulated  X',fontsize=25, family = 'Arial')
plt.ylabel('Simulated  Y', fontsize=25, family = 'Arial',labelpad=2)
plt.xticks([0,0.2,0.4,0.6,0.8,1.0],labels = ['0','0.2','0.4','0.6','0.8','1.0'],size = 25)
plt.yticks([0,0.2,0.4,0.6,0.8,1.0],labels = ['','0.2','0.4','0.6','0.8','1.0'],size = 25)
ax.set_xlim(0,1)                                  # 设置x坐标轴的显示范围
ax.set_ylim(0,1)                                  # 设置y坐标轴的显示范围

#内置plot绘制
axins = ax.inset_axes((0.3,0.3,0.7,0.7))
axins.scatter(x, y, marker='o', c=z, edgecolors=None ,s=15,cmap='Spectral_r')
axins.plot([0,1],[0,1],'black',lw=1)
axins.plot(x, y_est,'red',lw=1) #绘制拟合线
level = np.array([20,40,60,80]) #设置等高线特定值绘制
axins.tricontour(x, y, z,levels = level,cmap='Spectral_r',linewidths=1.0)

axins.set_xticks([0,0.1,0.2])
axins.set_xticklabels(['0','0.1','0.2'])
axins.set_yticks([0,0.1,0.2])
axins.set_yticklabels(['','0.1','0.2'])

axins.tick_params(axis='both',labelsize = 25)
axins.set_xlim(0, 0.3)
axins.set_ylim(0, 0.3)

# plt.savefig('C:/Users/dd/Desktop/hydrology/pearson.jpg',dpi=300,bbox_inches='tight',pad_inches=0)
plt.show()