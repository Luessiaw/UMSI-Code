# 地磁场梯度的容忍度
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
# paras.dipoleThetaRange = np.array([np.pi/3,np.pi/3])
paras.dipolePhiRange = np.array([0,0]) # 位置固定

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
theta = 0 # 主磁场方向沿 z 轴
paras.GeoFieldAtRef = 5e-5*(unit_x*np.sin(theta)+unit_z*np.cos(theta))
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
varName = "gradient"
refreshMode = 3 # 不需要重新计算 L,W
xs = np.linspace(0,500,11)/1e7 #单位：T/m
xticks = xs*1e7 # 单位：nT/cm

def varFunc3v(x,baseParas:Paras):
    return baseParas

def varFunc1(x,baseParas:Paras):
    newParas = deepcopy(baseParas)
    tensor = np.zeros((3,3))
    tensor[2,2] = x
    newParas.GeoFieldGradientTensor = tensor
    return newParas

def varFunc2(x,baseParas:Paras):
    newParas = deepcopy(baseParas)
    tensor = np.zeros((3,3))
    tensor[0,2] = x
    tensor[2,0] = x
    newParas.GeoFieldGradientTensor = tensor
    return newParas

def varFunc3(x,baseParas:Paras):
    newParas = deepcopy(baseParas)
    tensor = np.zeros((3,3))
    tensor[0,0] = x
    newParas.GeoFieldGradientTensor = tensor
    return newParas

saveFolder = "figs-original/data"
paras2v,paras2s,paras3v,paras3s = paras.childParas(numOfChannelsForDim2=15,
                                   numOfChannelsForDim3=64)

parass = []
varFuncs = []

for (i,thetaRange) in enumerate([0,np.pi/3]):
    paras3v.varFunc = varFunc3v
    paras3v.dipoleThetaRange = np.array([thetaRange,thetaRange])
    parass.append(paras3v)

    paras3s.dipoleThetaRange = np.array([thetaRange,thetaRange])
    paras3s1 = deepcopy(paras3s)
    paras3s1.labelPostfix = f"-Bzz-{i+1}"
    paras3s1.varFunc = varFunc1
    parass.append(paras3s1)

    paras3s2 = deepcopy(paras3s)
    paras3s2.labelPostfix = f"-Bzx-{i+1}"
    paras3s2.varFunc = varFunc2
    parass.append(paras3s2)

    paras3s3 = deepcopy(paras3s)
    paras3s3.labelPostfix = f"-Bxx-{i+1}"
    paras3s3.varFunc = varFunc3
    parass.append(paras3s3)

threads = []
vcs = []
for paras in parass:
    vc = VarContraller(varName,xs,paras.varFunc,paras,refreshMode)
    vcs.append(vc)
    thread = threading.Thread(target=vc.run,kwargs={"saveFolder":saveFolder})
    thread.start()
    threads.append(thread)
for thread in threads:
    thread.join()

for vc in vcs:
    txt = vc.getTXT()
    filename = os.path.join(saveFolder,f"fig5b-{vc.baseParas.getLabel()}.csv")
    with open(filename,"w",encoding="utf-8") as file:
        file.write(txt)
