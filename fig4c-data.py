import matplotlib
matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *

paras = Paras()

paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 0.8e-2

paras.dipoleStrength = 10e-9
# paras.dipoleRadiusRange = np.array([8e-2,8e-2]) # 固定深度
# paras.dipoleThetaRange = np.array([0,np.pi/3])
paras.dipolePhiRange = np.array([0,0])

paras.radiusOfSensorShell = 11e-2
paras.intrisicNoise = 100e-15
paras.externalNoise = 0

paras.GeoRefPos = origin
# theta = np.pi/4
# paras.GeoFieldAtRef = 5e-5*(unit_x*np.cos(theta)+unit_z*np.sin(theta))
# paras.GeoFieldAtRef = 5e-5*unit_z
paras.GeoFieldGradientTensor = np.zeros((3,3))

paras.regularPara = 1e-4
paras.threshold = 0.5

paras.numOfTrials = 500
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = None

rB = paras.radiusOfBrain

varName1 = "z"
xs1 = np.linspace(0,rB,12)
xticks1 = xs1*1e2
xticks1 = [f"{x:.1f}" for x in xticks1]
xlabel1 = "$z$ (cm)"

varName2 = "x"
xs2 = np.linspace(-rB,rB,26)
xticks2 = xs2*1e2
xticks2 = [f"{x:.0f}" for x in xticks2]
xlabel2 = "$x$ (cm)"

def varFunc(x1,x2,baseParas:Paras):
    newParas = deepcopy(baseParas)
    theta = np.arctan2(x2,x1)
    r = np.sqrt(x2**2+x1**2)
    newParas.dipoleRadiusRange = np.array([r,r])
    newParas.dipoleThetaRange = np.array([theta,theta])
    return newParas

saveFolder = "figs-original/data"

paras2v,paras2s,paras3v,paras3s = paras.childParas(numOfChannelsForDim2=15,
                                numOfChannelsForDim3=64)

ps = []
ps.append(paras3v)

thetas = [0,45,90]
for theta in thetas:
    thetaRAD = theta/180*np.pi
    paras3s.GeoFieldAtRef = 5e-5*(unit_x*np.sin(thetaRAD)+unit_z*np.cos(thetaRAD))
    paras3s.labelPostfix = str(theta)

    paras3sW = deepcopy(paras3s)
    paras3sW.considerDeadZone = True
    paras3sW.deadZoneType = "worst"
    ps.append(paras3sW)

    paras3sB = deepcopy(paras3s)
    paras3sB.considerDeadZone = True
    paras3sB.deadZoneType = "best"
    ps.append(paras3sB)

npzNames = []
for paras in ps:
    npzName = f"fig4c-{paras.getLabel()}.npz"
    npzNames.append(npzName)

def runBiVarControllers(varName1,xs1,
                        varName2,xs2,
                        varFunc,paras:Paras=None,
                        saveFolder="",
                        refreshMode = 0
                      ):
    vc = BiVarContraller(varName1,varName2,xs1,xs2,varFunc,paras)
    if refreshMode:
        vc.refreshMode = refreshMode
    vc.run(saveFolder=saveFolder)
    npzName = os.path.join(saveFolder,f"fig4c-{paras.getLabel()}.npz")
    np.savez(npzName,xs1=xs1,xs2=xs2,errs=vc.errs)

if __name__ == '__main__':
    # generate data
    processes = []
    for paras in ps:
        p = multiprocessing.Process(
            target=runBiVarControllers,
            args=(
                varName1,xs1,
                varName2,xs2,
                varFunc,paras,
            ),
            kwargs={
                "saveFolder":saveFolder,
                "refreshMode": 3
            }
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()



