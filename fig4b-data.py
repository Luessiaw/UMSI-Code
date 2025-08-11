# 地磁场方向的误差
import matplotlib
matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *

paras = Paras()

# paras.dim
paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 0.8e-2

paras.dipoleStrength = 10e-9
paras.dipoleRadiusRange = np.array([8e-2,8e-2]) # 位置固定
paras.dipoleThetaRange = np.array([0,0]) 
paras.dipolePhiRange = np.array([0,0])

# paras.sensorType
# paras.numOfChannels
paras.radiusOfSensorShell = 11e-2
paras.intrisicNoise = 100e-15
paras.externalNoise = 0
paras.considerDeadZone = False
paras.deadZoneType = "best" # best, worst, random
paras.axisAngleError = 0 
paras.considerRegistrate = False
paras.registrateType = "best"
paras.registrateError = 0

paras.GeoRefPos = origin
# theta = np.pi/2
theta = 0 # 主磁场方向沿 z 轴
paras.GeoFieldAtRef = 5e-5*(unit_x*np.sin(theta)+unit_z*np.cos(theta))
# paras.GeoFieldAtRef = None
paras.GeoFieldGradientTensor = np.zeros((3,3))
paras.GeoFieldGradientKnown = False

paras.regularPara = 1e-4
paras.threshold = 0.5

paras.numOfTrials = 500
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = None
# paras.labelPostfix

# 主磁场方向沿 z 轴
varName = "delta-phi"
refreshMode = 3 # 不需要重新计算 L,W
xs = np.linspace(0,np.pi/2,21) #单位：T/m
xticks = xs*180/np.pi # 单位：nT/cm

def varFunc(x,baseParas:Paras):
    newParas = deepcopy(baseParas)
    theta = newParas.theta + x
    newParas.theta = theta
    newParas.RealGeoFieldAtRef = 5e-5*(unit_x*np.sin(theta)+unit_z*np.cos(theta))
    return newParas

saveFolder = "./figs-original/data"

paras2v,paras2s,paras3v,paras3s = paras.childParas(numOfChannelsForDim2=15,
                                   numOfChannelsForDim3=64)

parass = []
paras3v.theta = 0
parass.append(paras3v)

paras3s.theta = 0
paras3s.GeoFieldAtRef = 5e-5*(unit_x*np.sin(paras3s.theta)+unit_z*np.cos(paras3s.theta))
parass.append(paras3s)

vcs = []
threads = []
for paras in parass:
    vc = VarContraller(varName,xs,varFunc,paras,refreshMode=refreshMode)
    vcs.append(vc)
    thread = threading.Thread(target=vc.run,kwargs={"saveFolder":saveFolder})
    thread.start()
    threads.append(thread)
for thread in threads:
    thread.join()

for vc in vcs:
    txt = vc.getTXT()
    filename = os.path.join(saveFolder,f"fig4b-{vc.baseParas.getLabel()}.csv")
    with open(filename,"w",encoding="utf-8") as file:
        file.write(txt)




