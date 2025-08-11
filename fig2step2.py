# 噪声
# import matplotlib
# matplotlib.use("Agg") 
from myMNE import *

shapes = ["o","d","s","v"]
colors = ["olivedrab","coral","steelblue","hotpink"]

paras = Paras()

paras.dim = 3
paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 0.8e-2

paras.dipoleStrength = 100e-9
# paras.dipoleRadiusRange = np.array([8e-2,10e-2]) 
# paras.dipoleThetaRange = np.array([0,np.pi/3]) 
# paras.dipolePhiRange = np.array([0,0])

paras.sensorType = "scalar"
paras.numOfChannels = 128
paras.radiusOfSensorShell = 11e-2
paras.intrisicNoise = 10e-15
paras.externalNoise = 0
paras.considerDeadZone = False
paras.deadZoneType = "best" # best, worst, random
paras.axisAngleError = 0 
paras.considerRegistrate = False
paras.registrateType = "best"
paras.registrateError = 0

paras.GeoRefPos = origin
theta = 0 
paras.GeoFieldAtRef = 5e-5*(unit_x*np.sin(theta)+unit_z*np.cos(theta))
paras.GeoFieldGradientTensor = np.zeros((3,3))
paras.GeoFieldGradientKnown = True

paras.regularPara = 1e-4
paras.threshold = 0.5

paras.numOfTrials = 1
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = (np.array([0.5,0,7.5])/100,np.array([0,10,0])*1e-9)


sol = Solver(paras)
sensorPoints = sol.sensorPoints
trial = sol.singleTrial()

# fig2a
ps = sol.sensorPoints*1e2
fig = vs.plt.figure()
ax = fig.add_subplot(1,1,1,projection="3d")
ax.scatter(ps[0,:],ps[1,:],ps[2,:],c="blue",s=30)
ax.set_axis_off()
ax.set_box_aspect((1,1,0.5))
fig.savefig("./figs-original/subplots/fig2a.png",bbox_inches='tight', pad_inches=0, transparent=True)


# fig2c
xv,yv,Bm = sol.getTheoBm(trial.rp,trial.p,num=100)
xv *= 1e2
yv *= 1e2
Bm *= 1e12
norm = vs.plt.Normalize(vmin=Bm.min(), vmax=Bm.max())
cmap = vs.cm.rainbow  
colors = cmap(norm(Bm))  

xs = sol.sensorPoints[0,:]*1e2
ys = sol.sensorPoints[1,:]*1e2

fig = vs.plt.figure()
ax = fig.add_subplot(1,1,1)
ax.pcolormesh(xv,yv,Bm,cmap=cmap,norm=norm)
ax.scatter(xs,ys,s=15,c="gray")
ax.set_aspect("equal")
ax.set_axis_off()

fig.savefig("./figs-original/subplots/fig2c.png",bbox_inches='tight', pad_inches=0, transparent=True)

# fig2de 
ps = sol.sourcePoints*1e2
Q = trial.Q
Q1 = Q[:sol.numOfSourcePoints]**2 + Q[sol.numOfSourcePoints:]**2
amplitude = Q1/np.max(Q1)

fig = vs.plt.figure()
ax = fig.add_subplot(1,1,1,projection="3d")
vs.plotSphere(origin,sol.paras.radiusOfHead*1e2,ax=ax,color="lightgray",alpha=0.05)
ax.scatter(
    ps[0,:],ps[1,:],ps[2,:],s=30,c=amplitude,cmap="Reds"
)
ax.set_box_aspect([1,1,1])
ax.set_axis_off()

fig.savefig("./figs-original/subplots/fig2de.png",bbox_inches='tight', pad_inches=0, transparent=True)




