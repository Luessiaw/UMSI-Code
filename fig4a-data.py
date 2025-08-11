# 地磁场方向
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
# theta = np.pi/2
# theta = 0 # 主磁场方向沿 z 轴
# paras.GeoFieldAtRef = 5e-5*(unit_x*np.sin(theta)+unit_z*np.cos(theta))
paras.GeoFieldAtRef = None
paras.GeoFieldGradientTensor = np.zeros((3,3))
paras.GeoFieldGradientKnown = True

paras.regularPara = 1e-4
paras.threshold = 0.5

paras.numOfTrials = 500
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = None
# paras.labelPostfix

# 主磁场方向沿 z 轴
varName = "geo-ori"
refreshMode = 1
xs = np.linspace(0,np.pi/2,20)
xticks = xs*180/np.pi

def varFunc(x,baseParas:Paras):
    newParas = deepcopy(baseParas)
    newParas.GeoFieldAtRef = 5e-5*(unit_x*np.sin(x)+unit_z*np.cos(x))
    return newParas

saveFolder = "./figs-original/data"

paras2v,paras2s,paras3v,paras3s = paras.childParas(numOfChannelsForDim2=15,
                                   numOfChannelsForDim3=64)

parass = []
parass.append(paras3v)

paras3sB = deepcopy(paras3s)
paras3sB.considerDeadZone = False
paras3sB.deadZoneType = "best"
parass.append(paras3sB)

paras3sW = deepcopy(paras3s)
paras3sW.considerDeadZone = True
paras3sW.deadZoneType = "worst"
parass.append(paras3sW)

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
    # vc.run()

for i,vc in enumerate(vcs):
    txt = vc.getTXT()
    filename = os.path.join(saveFolder,f"fig4a-{vc.baseParas.getLabel()}.csv")
    with open(filename,"w",encoding="utf-8") as file:
        file.write(txt)

