import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#import pandas as pd
import torch
import numpy as np
np.random.seed(42)
from matplotlib import pyplot as plt
from copy import deepcopy
from mp_pytorch.mp import MPFactory
from faprodmp.utils import cholesky

traj_label = 'dataset5'
fig_dir = 'fig5'
os.makedirs(fig_dir, exist_ok=True)
nbDemos = 5
T = 100 #110#106#83#57
frame_indices = [0, -1]
nbFrames = len(frame_indices)
if traj_label != 'dataset6':
    POSITION_FEATURES = [ 'px', 'py', 'pz']
else:
    POSITION_FEATURES = [ 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7' ]
nbStates = len(POSITION_FEATURES)
demo_dir = "arms_demos/D1/"
TAU = 1 # prerequisite: recorded data index has already been normalized in range [0, 1]
DT = 1e-3 # precision of the precomputation
AUTO_SCALE = False
REGULARIZATION_FACTOR = 1e-9
DTYPE = torch.float64
LEARN_TAU = False
LEARN_DELAY = False
BASIS_BANDWIDTH_FACTOR = 2
NUM_BASIS_OUTSIDE = 0
ALPHA = 100
ALPHA_PHASE = 2 
NUM_BASIS = 10
STD_SCALE = 2
mp_config = dict()
mp_config["mp_type"] = 'prodmp'
mp_config["num_dof"] = len(POSITION_FEATURES)
mp_config["tau"] = TAU
mp_config["dtype"] = DTYPE
mp_config["learn_tau"] = LEARN_TAU
mp_config["learn_delay"] = LEARN_DELAY
mp_config["mp_args"] = dict()
mp_config["mp_args"]["dt"] = DT
mp_config["mp_args"]["num_basis"] = NUM_BASIS
mp_config["mp_args"]["basis_bandwidth_factor"] = BASIS_BANDWIDTH_FACTOR
mp_config["mp_args"]["num_basis_outside"] = NUM_BASIS_OUTSIDE
mp_config["mp_args"]["alpha"] = ALPHA
mp_config["mp_args"]["alpha_phase"] = ALPHA_PHASE
mp_config["mp_args"]["auto_scale_basis"] = AUTO_SCALE



# --- load demos ---
urdf_path = r"urdf_files_dataset\urdf_files\oems\xacro_generated\franka_emika\franka_description\robots\panda\panda.urdf"
import pinocchio as pin
import numpy as np
from pathlib import Path

def get_ee_trajectory(urdf_path: str, joint_trajectory: np.ndarray, ee_frame_name: str = "panda_link7"):
    """
    Compute end-effector trajectory from joint positions.

    Args:
        urdf_path:        path to franka_description URDF
        joint_trajectory: (T, nq) array of joint positions [rad]
        ee_frame_name:    name of the EE frame in the URDF

    Returns:
        positions:    (T, 3)    EE positions in world frame
        orientations: (T, 3, 3) EE rotation matrices in world frame
    """
    # Load model
    model = pin.buildModelFromUrdf(urdf_path)
    data  = model.createData()

    # Get the frame ID for the end effector
    ee_frame_id = model.getFrameId(ee_frame_name)
    if ee_frame_id == model.nframes:
        available = [model.frames[i].name for i in range(model.nframes)]
        raise ValueError(f"Frame '{ee_frame_name}' not found. Available frames:\n{available}")

    positions    = []
    orientations = []

    for q in joint_trajectory:
        q_pin = np.array(q, dtype=np.float64)

        # Forward kinematics
        pin.forwardKinematics(model, data, q_pin)
        pin.updateFramePlacement(model, data, ee_frame_id)

        T = data.oMf[ee_frame_id]   # SE3 transform: world <- EE
        positions.append(T.translation.copy())
        orientations.append(T.rotation.copy())

    return np.array(positions), np.array(orientations)

def dataset1(nbDemos=1,seed=0,starts=None, ends=None):
    # Different start and end points per demo
    rng = np.random.default_rng(seed)
 
    if starts is not None:
        starts = starts
    else:
        starts = rng.normal(loc=np.array([-0.6, -0.2, -0.6]), scale=0.05, size=(nbDemos, 3))
    if ends is not None:
        ends = ends
    else:
        ends = rng.normal(loc=np.array([ 0.4,  0.2,  0.4]),   scale=0.5,   size=(nbDemos, 3))
    
    trajectories_list = []
    for n in range(nbDemos):
        start = starts[n]   # (3,)
        end   = ends[n]     # (3,)
    
        # linear interpolation scaffold from start to end
        blend = np.linspace(0, 1, T)[:, None]  # (T, 1)
        linear = (1 - blend) * start + blend * end  # (T, 3)
    
        # add the base shape as a deviation on top of the linear path
        # base is already zero at t=0 and t=1 (y arc), and zero-mean for x/z after subtracting linear
        # so we just add the arc component to give it a 3D shape
        arc = np.zeros((T, 3))
        arc[:, 1] = np.sin(np.pi * np.linspace(0, 1, T)) * 0.15  # lift in y
        arc[:, 0] = -0.1 * np.sin(2 * np.pi * np.linspace(0, 1, T))  # S in x
        arc[:, 2] =  0.1 * np.sin(2 * np.pi * np.linspace(0, 1, T))  # S in z
    
        traj = linear + arc
    
        # small noise
        noise = np.random.normal(0, 0.005, size=traj.shape)
        traj += noise
    
        trajectories_list.append(traj)

    return trajectories_list

def dataset2(nbDemos=1,seed=0,starts=None, ends=None):
    t = np.linspace(0, 1,T)
    window = t * (1 - t) * 4  # bell: 0 at edges, 1 at middle
 
    def make_trajectory(start, end, noise_std=0.005):
        blend = t[:, None]
        linear = (1 - blend) * start + blend * end
    
        deviation = np.zeros((T, 3))
        # sharp dip DOWN first, then big swing UP and over
        deviation[:, 1] = (-0.4 * np.sin(np.pi * t)        # main arc up
                           + 0.15 * np.sin(2 * np.pi * t)) # dip at start
        # corkscrew: x and z rotate around the straight-line axis
        deviation[:, 0] = 0.35 * np.sin(1.5 * np.pi * t) * window
        deviation[:, 2] = 0.35 * np.cos(1.5 * np.pi * t) * window - 0.35 * window
    
        traj = linear + deviation
        traj += np.random.normal(0, noise_std, size=traj.shape)
        return traj
    
    rng = np.random.default_rng(seed)
    
    if starts is not None:
        starts = starts
    else:
        starts = rng.normal(loc=np.array([-0.6, -0.2, -0.6]), scale=0.05, size=(nbDemos, 3))
    if ends is not None:
        ends = ends
    else:
        ends = rng.normal(loc=np.array([ 0.4,  0.2,  0.4]),   scale=0.5,   size=(nbDemos, 3))
    
    trajectories_list = [
        make_trajectory(starts[n], ends[n]) for n in range(nbDemos)
    ]
    return trajectories_list

def dataset3(nbDemos=1,seed=0,starts=None,ends=None):
    #starts = np.array([-0.8,0,-0.8])
    #ends = np.array([-0.25, 2, 0.25])
    Demo1 = np.array([
                  [-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80985,-0.80972,-0.80786,-0.83445,-0.8132,-0.81441,-0.81481,-0.81513,-0.81546,-0.81578,-0.81611,-0.81643,-0.81675,-0.81708,-0.8174,-0.81773,-0.81805,-0.81837,-0.8187,-0.81902,-0.81934,-0.81967,-0.81999,-0.82032,-0.82064,-0.82096,-0.82129,-0.82161,-0.82193,-0.82227,-0.82272,-0.82491,-0.79864,-0.82022,-0.81932,-0.81925,-0.81925,-0.81925,-0.81925,-0.81925,-0.81931,-0.82021,-0.80703,-0.81754,-0.81697,-0.81678,-0.81661,-0.81645,-0.81627,-0.81584,-0.81033,-0.80999,-0.81318,-0.81305,-0.81274,-0.81242,-0.8121,-0.81177,-0.81145,-0.81113,-0.81083,-0.81078,-0.81417,-0.82655,-0.83864,-0.81966,-0.83972,-0.8448,-0.85032,-0.86506,-0.86179,-0.88087,-0.87707,-0.88397,-0.87058,-0.87344,-0.87663,-0.87988,-0.88339,-0.89197,-0.89543,-0.89598,-0.87818,-0.89615,-0.90026,-0.89638,-0.89647,-0.89071,-0.8874,-0.89579,-0.89603,-0.88084,-0.85869,-0.84715,-0.87092,-0.85941,-0.842,-0.8308,-0.8257,-0.81549,-0.7982,-0.76986,-0.74871,-0.74382,-0.75211,-0.75182,-0.73802,-0.71597,-0.70285,-0.70013,-0.68668,-0.7059,-0.68011,-0.67917,-0.67737,-0.6716,-0.6717,-0.66934,-0.65526,-0.64908,-0.65487,-0.65838,-0.64369,-0.62361,-0.57827,-0.58402,-0.5506,-0.54803,-0.54763,-0.55523,-0.56753,-0.5567,-0.57866,-0.53611,-0.52262,-0.50107,-0.52377,-0.52089,-0.49962,-0.47864,-0.48467,-0.46316,-0.45836,-0.44697,-0.46494,-0.44282,-0.43484,-0.42721,-0.4196,-0.41201,-0.40467,-0.4052,-0.18722,-0.43164,-0.44266,-0.37995,-0.37619,-0.39327,-0.46838,-0.41842,-0.53138,-0.54188,-0.46836,-0.37397,-0.42476,-0.42488,-0.42488,-0.42488],
                  0.01*np.arange(200)-1,
                  [-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79733,-0.79732,-0.79729,-0.79709,-0.77705,-0.79341,-0.78235,-0.78108,-0.78804,-0.76739,-0.78277,-0.76255,-0.75618,-0.76322,-0.7481,-0.74991,-0.73962,-0.75021,-0.7436,-0.72405,-0.73088,-0.71262,-0.72347,-0.71851,-0.69678,-0.70199,-0.69559,-0.6907,-0.6759,-0.67122,-0.67394,-0.66136,-0.65514,-0.68066,-0.6467,-0.65136,-0.64539,-0.63091,-0.6427,-0.62181,-0.63369,-0.6217,-0.60373,-0.61913,-0.59836,-0.59746,-0.58644,-0.58246,-0.59212,-0.58694,-0.58508,-0.56703,-0.56098,-0.58097,-0.54849,-0.5543,-0.54338,-0.54957,-0.54321,-0.52801,-0.53667,-0.52275,-0.49651,-0.49414,-0.5061,-0.48612,-0.4926,-0.47562,-0.46672,-0.45962,-0.46202,-0.45399,-0.42765,-0.42211,-0.4178,-0.39077,-0.38175,-0.38577,-0.3643,-0.34443,-0.33927,-0.29452,-0.29986,-0.27169,-0.23172,-0.24448,-0.21529,-0.15336,-0.13499,-0.14264,-0.11548,-0.11546,-0.1129,-0.079211,-0.048828,-0.047322,-0.022471,0.016414,0.010184,0.037657,0.046362,0.067063,0.056725,0.10176,0.10241,0.13035,0.14679,0.14856,0.13821,0.15903,0.13658,0.17351,0.1514,0.14417,0.17445,0.16998,0.13034,0.13277,0.16931,0.17734,0.19328,0.18754,0.2131,0.17394,0.22603,0.19314,0.17443,0.1998,0.19091,0.19776,0.18486,0.21725,0.18082,0.19924,0.189,0.19044,0.24204,0.26059,0.24735,0.25459,0.2482,0.27198,0.25184,0.28322,0.26596,0.27956,0.28626,0.30924,0.27589,0.28617,0.30221,0.31971,0.33286,0.29727,0.3203,0.32513,0.32989,0.33466,0.33946,0.34494,0.3377,0.35292,0.3457,0.34771,0.48341,0.2935,0.3529,0.35997,0.36766,0.39986,0.33718,0.37462,0.2002,0.30964,0.34333,0.34299,0.34298,0.34298,0.34298,0.34298]]).T

    Demo2 = np.array([
                  [-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80516,-0.80517,-0.80514,-0.80525,-0.80495,-0.80562,-0.80467,-0.79508,-0.80289,-0.79574,-0.79107,-0.78849,-0.79125,-0.79423,-0.78143,-0.77167,-0.77655,-0.77478,-0.76159,-0.76172,-0.76221,-0.75737,-0.75297,-0.75082,-0.75566,-0.75264,-0.75056,-0.74609,-0.75711,-0.77159,-0.77709,-0.78156,-0.78329,-0.78013,-0.7926,-0.79888,-0.81534,-0.80608,-0.81637,-0.82128,-0.82483,-0.83594,-0.83684,-0.84682,-0.85195,-0.84903,-0.85419,-0.86916,-0.87446,-0.86872,-0.86163,-0.87621,-0.87791,-0.8641,-0.8656,-0.85929,-0.85204,-0.82251,-0.78281,-0.76111,-0.75308,-0.75556,-0.76883,-0.72938,-0.72733,-0.70291,-0.6984,-0.7006,-0.67146,-0.65005,-0.59471,-0.57786,-0.58871,-0.58404,-0.56421,-0.54959,-0.52154,-0.49059,-0.50438,-0.49838,-0.47169,-0.46648,-0.45372,-0.44167,-0.44074,-0.4361,-0.43951,-0.45016,-0.45071,-0.41921,-0.41039,-0.39291,-0.38327,-0.36914,-0.38625,-0.36406,-0.34505,-0.36565,-0.37041,-0.39734,-0.38774,-0.35738,-0.35806,-0.3575,-0.33508,-0.34524,-0.35974,-0.33434,-0.31363,-0.32191,-0.30776,-0.30684,-0.30939,-0.29402,-0.28348,-0.28668,-0.28108,-0.26905,-0.26961,-0.26099,-0.25618,-0.26824,-0.25797,-0.23702,-0.24296,-0.22463,-0.21853,-0.20411,-0.1987,-0.19818,-0.18864,-0.17436,-0.16565,-0.16423,-0.15167,-0.15167,-0.14829,-0.15162,-0.13635,-0.13165,-0.12111,-0.116,-0.1225,-0.12476,-0.11212,-0.10791,-0.11111,-0.12122,-0.1099,-0.097918,-0.097373,-0.094037,0.030096,-0.057334,-0.051302,-0.048413,-0.072396,-0.13781,-0.1001,-0.1156,-0.15834,-0.15735,-0.095541,-0.096259,-0.096257,-0.096246,-0.096244,-0.096244],
                  0.01*np.arange(200)-1,
                  [-0.815,-0.81431,-0.81379,-0.68562,-0.6931,-0.64064,-0.60548,-0.69295,-0.56933,-0.63964,-0.66578,-0.60926,-0.61126,-0.79019,-0.62204,-0.6112,-0.58727,-0.57014,-0.57389,-0.56784,-0.55393,-0.53696,-0.55201,-0.50981,-0.50837,-0.51129,-0.49489,-0.49953,-0.50191,-0.461,-0.46059,-0.43567,-0.4708,-0.45102,-0.43147,-0.41375,-0.40033,-0.40387,-0.41176,-0.4023,-0.39729,-0.36936,-0.34603,-0.33549,-0.32438,-0.32538,-0.304,-0.28347,-0.26307,-0.25865,-0.24439,-0.23184,-0.24339,-0.22772,-0.20781,-0.20025,-0.19386,-0.16423,-0.18397,-0.15952,-0.15421,-0.11471,-0.10395,-0.11295,-0.098239,-0.093953,-0.073039,-0.064267,-0.055597,-0.031172,-0.035399,-0.023356,-0.014331,-0.00019282,0.016371,-0.0049711,0.05631,0.058131,0.086297,0.077463,0.099461,0.11885,0.11167,0.14594,0.13516,0.15271,0.17467,0.18453,0.19857,0.20708,0.19936,0.22216,0.24739,0.2431,0.26444,0.26652,0.29121,0.30261,0.30743,0.30966,0.31298,0.33155,0.31944,0.31972,0.34268,0.30411,0.31498,0.30974,0.32343,0.32662,0.3226,0.34209,0.32413,0.3578,0.37109,0.34259,0.34613,0.34826,0.34623,0.36097,0.34305,0.36196,0.35324,0.37377,0.37647,0.36638,0.38104,0.37966,0.37867,0.38209,0.36891,0.38656,0.38978,0.38775,0.40449,0.40807,0.40853,0.44288,0.41424,0.41601,0.44883,0.43744,0.42993,0.43156,0.44545,0.4612,0.45259,0.47036,0.47283,0.4877,0.48428,0.50009,0.47522,0.49133,0.49933,0.48004,0.50778,0.52892,0.51687,0.53313,0.55168,0.54379,0.55238,0.5594,0.56763,0.59655,0.56098,0.5815,0.60546,0.57505,0.60061,0.60991,0.59352,0.60276,0.60239,0.62066,0.6064,0.59913,0.60848,0.60115,0.60944,0.62089,0.61881,0.6251,0.73426,0.64059,0.65271,0.6631,0.63333,0.53316,0.61336,0.59235,0.56279,0.62217,0.61936,0.61915,0.61915,0.61915,0.61915,0.61915]]).T

    Demo3 = np.array([
                  [-0.79577,-0.79577,-0.79577,-0.79577,-0.79577,-0.79577,-0.79577,-0.79577,-0.79577,-0.79577,-0.79577,-0.79578,-0.79577,-0.79581,-0.7956,-0.79599,-0.79625,-0.79355,-0.79778,-0.81063,-0.80781,-0.79824,-0.79781,-0.80876,-0.81739,-0.80874,-0.8046,-0.80759,-0.80822,-0.80844,-0.80925,-0.81001,-0.81041,-0.81098,-0.81299,-0.81129,-0.80557,-0.80801,-0.81323,-0.81367,-0.80997,-0.80532,-0.80379,-0.80688,-0.81052,-0.8103,-0.81261,-0.80878,-0.7961,-0.79904,-0.80878,-0.80938,-0.79858,-0.79012,-0.79892,-0.80324,-0.80044,-0.79979,-0.79994,-0.79987,-0.7969,-0.79839,-0.81117,-0.80971,-0.80155,-0.80764,-0.80734,-0.802,-0.80192,-0.80638,-0.81151,-0.81206,-0.7998,-0.7779,-0.76978,-0.76669,-0.75366,-0.71423,-0.66906,-0.64841,-0.63587,-0.62176,-0.6042,-0.60485,-0.60409,-0.55056,-0.52091,-0.53,-0.49529,-0.46244,-0.45482,-0.42915,-0.4034,-0.37922,-0.34889,-0.31403,-0.27405,-0.26343,-0.26328,-0.2318,-0.20102,-0.18633,-0.17703,-0.15811,-0.1394,-0.13799,-0.15363,-0.16102,-0.15349,-0.13823,-0.11721,-0.10763,-0.082832,-0.053788,-0.07651,-0.077231,-0.042012,-0.053211,-0.063311,-0.0349,-0.016891,-0.019198,-0.030413,-0.034817,-0.039686,-0.053838,-0.034796,-0.00057366,-0.0059988,-0.015169,-0.011834,-0.0077093,-0.0020358,0.0059312,0.014328,0.015846,0.022046,0.034262,0.035624,0.026378,0.031757,0.046005,0.046798,0.055033,0.074104,0.067476,0.06364,0.081194,0.074423,0.066839,0.082196,0.094266,0.10027,0.1097,0.12427,0.13388,0.13187,0.13673,0.14493,0.14509,0.14676,0.15371,0.15612,0.1711,0.19348,0.18212,0.16635,0.16747,0.17742,0.18565,0.18996,0.19108,0.17749,0.1638,0.18038,0.19524,0.18699,0.19406,0.2056,0.20166,0.20109,0.20386,0.2009,0.1947,0.2436,0.19523,0.23399,0.22857,0.19729,0.15972,0.22719,0.1759,0.14522,0.20466,0.19865,0.19952,0.19958,0.19952,0.19953,0.19953],
                  0.01*np.arange(200)-1,
                  [-0.78809,-0.78984,-0.78844,-0.75376,-0.68074,-0.7436,-0.72983,-0.70441,-0.72582,-0.71577,-0.72971,-0.73162,-0.69148,-0.75376,-0.69395,-0.6908,-0.69472,-0.68763,-0.67443,-0.6632,-0.65566,-0.64418,-0.63469,-0.64877,-0.64626,-0.6165,-0.61535,-0.62137,-0.61464,-0.6064,-0.59565,-0.59271,-0.58512,-0.57459,-0.57352,-0.56607,-0.55378,-0.54554,-0.52862,-0.50696,-0.51443,-0.52517,-0.51602,-0.5115,-0.502,-0.48005,-0.46918,-0.46832,-0.47131,-0.46349,-0.44591,-0.43399,-0.41609,-0.40033,-0.41275,-0.40902,-0.36974,-0.35325,-0.35484,-0.34159,-0.31775,-0.30322,-0.30138,-0.28796,-0.26189,-0.23686,-0.23013,-0.24243,-0.2459,-0.22235,-0.19249,-0.18096,-0.16537,-0.14354,-0.13999,-0.13793,-0.119,-0.089394,-0.075255,-0.08704,-0.084541,-0.077259,-0.083048,-0.057308,-0.033048,-0.056342,-0.055451,-0.038351,-0.044531,-0.049537,-0.04377,-0.030177,-0.029181,-0.041116,-0.040782,-0.026307,-0.017565,-0.012021,-0.0021467,0.006485,0.01773,0.019269,0.004092,0.016605,0.040489,0.019447,0.0050942,0.029083,0.053689,0.058654,0.06007,0.073254,0.076539,0.077206,0.095195,0.10963,0.11283,0.11462,0.11981,0.13423,0.1397,0.13779,0.14543,0.1434,0.14179,0.15933,0.1629,0.15352,0.16218,0.16913,0.16185,0.17647,0.18454,0.17723,0.19608,0.20477,0.19501,0.2013,0.21355,0.22375,0.23193,0.23562,0.24326,0.2583,0.26909,0.25415,0.25469,0.28931,0.28319,0.26963,0.28511,0.30097,0.31388,0.32134,0.33209,0.34785,0.35467,0.34784,0.34753,0.35586,0.35845,0.36355,0.36705,0.38008,0.40107,0.39974,0.39146,0.38872,0.38807,0.39497,0.40744,0.4081,0.39842,0.38807,0.40908,0.4238,0.4006,0.4077,0.43097,0.43426,0.42839,0.42337,0.42586,0.42359,0.48374,0.42574,0.46101,0.47256,0.4503,0.40078,0.43914,0.43311,0.35416,0.39392,0.43037,0.42766,0.42745,0.42765,0.42762,0.42762]]).T

    Demo4 = np.array([
                  [-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80986,-0.80987,-0.80983,-0.80991,-0.80997,-0.80819,-0.82475,-0.82236,-0.81384,-0.81332,-0.81552,-0.81553,-0.81554,-0.81602,-0.81636,-0.81665,-0.81699,-0.8173,-0.81761,-0.81802,-0.81828,-0.81828,-0.81935,-0.82023,-0.81603,-0.8123,-0.81606,-0.81921,-0.81846,-0.81832,-0.81888,-0.81848,-0.81876,-0.82142,-0.81772,-0.80627,-0.80665,-0.81602,-0.81779,-0.81191,-0.80702,-0.81082,-0.81264,-0.80576,-0.80488,-0.8095,-0.80639,-0.80008,-0.79577,-0.78618,-0.77304,-0.76829,-0.76721,-0.76568,-0.76355,-0.75178,-0.72342,-0.69334,-0.67684,-0.66682,-0.65874,-0.6496,-0.6373,-0.61911,-0.58704,-0.54259,-0.5162,-0.50631,-0.49727,-0.48929,-0.47084,-0.41834,-0.35577,-0.33037,-0.32087,-0.31049,-0.28715,-0.25008,-0.22631,-0.20724,-0.17142,-0.14767,-0.14094,-0.099274,-0.043401,-0.024489,-0.025004,-0.014856,0.02047,0.06428,0.089541,0.10362,0.11587,0.13679,0.15793,0.18393,0.21669,0.24072,0.26397,0.28412,0.26385,0.22848,0.2474,0.29211,0.3257,0.33126,0.32618,0.34245,0.35093,0.34211,0.36548,0.40411,0.39092,0.361,0.35905,0.3751,0.39313,0.38559,0.3731,0.38503,0.40682,0.423,0.41145,0.38356,0.36941,0.37252,0.38335,0.38601,0.39044,0.40791,0.42038,0.41798,0.41044,0.40899,0.42502,0.43064,0.41481,0.42226,0.44553,0.4436,0.43654,0.44848,0.46154,0.46526,0.46163,0.4584,0.46529,0.48075,0.48926,0.47734,0.46594,0.47075,0.482,0.49359,0.4967,0.49381,0.49665,0.50107,0.49974,0.49902,0.50165,0.50529,0.50533,0.49972,0.50354,0.51531,0.51252,0.50388,0.50383,0.50514,0.50418,0.50247,0.50193,0.50457,0.50822,0.50929,0.49995,0.49053,0.51662,0.52008,0.49078,0.48524,0.50562,0.50427,0.50481,0.50467,0.5047,0.5047,0.50469,0.50469,0.50469,0.50469,0.50469,0.50469],
                  0.01*np.arange(200)-1,
                  [-0.79288,-0.79274,-0.79341,-0.79135,-0.79442,-0.71115,-0.67289,-0.68134,-0.69919,-0.73853,-0.72031,-0.73016,-0.69903,-0.77298,-0.70614,-0.69658,-0.68581,-0.67889,-0.67168,-0.66338,-0.65493,-0.64638,-0.63793,-0.62833,-0.61742,-0.6118,-0.611,-0.60784,-0.60387,-0.59786,-0.58384,-0.56911,-0.56105,-0.55366,-0.54307,-0.53054,-0.52201,-0.51416,-0.50245,-0.49326,-0.49234,-0.49257,-0.47939,-0.46194,-0.45323,-0.44178,-0.42716,-0.4245,-0.42058,-0.40459,-0.39655,-0.39327,-0.37128,-0.35116,-0.35177,-0.33834,-0.31015,-0.30406,-0.30264,-0.28697,-0.27929,-0.27678,-0.25817,-0.23716,-0.23693,-0.23525,-0.21772,-0.21297,-0.21802,-0.20676,-0.19588,-0.20157,-0.20507,-0.20284,-0.20655,-0.20979,-0.20219,-0.21148,-0.23822,-0.23295,-0.22053,-0.24524,-0.2623,-0.24847,-0.25457,-0.28314,-0.29241,-0.28773,-0.29842,-0.32267,-0.34253,-0.36378,-0.37757,-0.37219,-0.38404,-0.40779,-0.40184,-0.39388,-0.41212,-0.42331,-0.41842,-0.41867,-0.42519,-0.43135,-0.43928,-0.44481,-0.42715,-0.40529,-0.41575,-0.41923,-0.3994,-0.40775,-0.4235,-0.40501,-0.39107,-0.3946,-0.4022,-0.40913,-0.39154,-0.36664,-0.35592,-0.3489,-0.34951,-0.35345,-0.3453,-0.33522,-0.32842,-0.31615,-0.3099,-0.30923,-0.30568,-0.29771,-0.2859,-0.27241,-0.26627,-0.27614,-0.27504,-0.25592,-0.25197,-0.25884,-0.24597,-0.22995,-0.23791,-0.24273,-0.2261,-0.205,-0.19905,-0.20416,-0.18329,-0.15817,-0.17554,-0.1861,-0.15437,-0.13025,-0.13482,-0.1464,-0.14424,-0.12763,-0.1215,-0.12066,-0.1026,-0.09291,-0.10791,-0.10988,-0.092293,-0.087489,-0.094409,-0.09479,-0.089442,-0.081949,-0.080441,-0.083767,-0.078693,-0.073356,-0.078244,-0.076129,-0.067214,-0.081864,-0.097967,-0.079456,-0.065634,-0.076466,-0.088125,-0.090929,-0.021674,-0.082117,-0.087621,-0.15199,-0.075628,-0.082252,-0.07963,-0.080284,-0.080167,-0.080176,-0.08018,-0.080178,-0.080178,-0.080178,-0.080178,-0.080178]]).T
    
    demos = np.array([Demo1, Demo2, Demo3, Demo4])
    def fit_latent_function(demos, latent_dim=2):
        """
        demos: (n_demos, 3, T)
        returns:
            mean_traj: (3, T)
            W: (latent_dim, 3*T)
            Z: (n_demos, latent_dim)
        """
        n_demos, D, T = demos.shape

        X = demos.reshape(n_demos, D * T)  # (n_demos, 3T)

        mean_traj = X.mean(axis=0, keepdims=True)
        X_centered = X - mean_traj

        # SVD (PCA)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        W = Vt[:latent_dim]              # (latent_dim, 3T)
        Z = X_centered @ W.T             # (n_demos, latent_dim)

        return mean_traj.reshape(D, T), W, Z
    def reconstruct_demo(mean_traj, W, z):
        D, T = mean_traj.shape
        mean_flat = mean_traj.reshape(-1)
        x = mean_flat + z @ W
        return x.reshape(D, T)
    def sample_latent(Z):
        z_mean = Z.mean(axis=0)
        z_cov = np.cov(Z.T)
        return np.random.multivariate_normal(z_mean, z_cov)
    def sample_demo(mean_traj, W, Z):
        z = sample_latent(Z)
        return reconstruct_demo(mean_traj, W, z)
    def sample_demos(mean_traj, W, Z, n_samples=5):
        return [sample_demo(mean_traj, W, Z)[::2] for _ in range(n_samples)] #np.stack([sample_demo(mean_traj, W, Z) for _ in range(n_samples)])
    def sample_demos_constrained(mean_traj, W, x_start, x_end, n_samples=5, eps=1e-6):

        T, D = mean_traj.shape
        mu = mean_traj.reshape(-1)

        # trajectory covariance induced by latent model
        Sigma_x = W.T @ W + eps * np.eye(T * D)

        # observation matrix (start + end)
        A = np.zeros((6, T * D))
        A[:3, :3] = np.eye(3)
        A[3:, -3:] = np.eye(3)

        y = np.hstack([x_start, x_end])

        # conditioning
        Sigma_y = A @ Sigma_x @ A.T + eps * np.eye(6)
        K = Sigma_x @ A.T @ np.linalg.inv(Sigma_y)

        mu_cond = mu + K @ (y - A @ mu)
        Sigma_cond = Sigma_x - K @ A @ Sigma_x

        # sample trajectories in flat space
        L = np.linalg.cholesky(Sigma_cond + eps * np.eye(T * D))

        samples = []
        for _ in range(n_samples):
            x = mu_cond + L @ np.random.randn(T * D)
            samples.append(x.reshape(T, D)[::2])

        return samples #np.stack(samples)

    mean_traj, W, Z = fit_latent_function(demos)
    # sampled new demos
    if starts is None:
        samples = sample_demos(mean_traj, W, Z, n_samples=nbDemos)
    else:
        samples = sample_demos_constrained(
    mean_traj=mean_traj,
    W=W,
    x_start=starts.squeeze(),
    x_end=ends.squeeze(),
    n_samples=nbDemos
)

    return samples

def dataset4(nbDemos=1,seed=0,starts=None,ends=None):
    trajectories_list = []
    t = np.linspace(0, 2 * np.pi,T)
    base_trajectory = np.column_stack([np.cos(t)+0., np.sin(t), t / (2 * np.pi)])
    for _ in range(nbDemos):
        noise = np.random.normal(0, 0.02, size=base_trajectory.shape)
        trajectories_list.append(base_trajectory + noise)
    
    return trajectories_list

def dataset5(nbDemos=1,seed=0,starts=None,ends=None):
    trajectories_list = []
    for i in range(nbDemos):
        if starts is not None:
            start_offset = starts
        else:
            start_offset = np.random.normal(0, 0.05, size=3)
        if ends is not None:
            goal_offset = ends
        else:
            goal_offset = np.random.normal(-0.5, 0.5, size=3)

        # 3. Random speed/timing variation (stretch/compress trajectory)
        speed = 1#np.random.uniform(0.8, 1.2)
        t_demo = np.linspace(0, 2 * np.pi * speed, T)

        # 4. Random amplitude variation
        amp_x = np.random.uniform(0.8, 1.2)
        amp_y = np.random.uniform(0.8, 1.2)

        base = np.column_stack([
            amp_x * np.cos(t_demo),
            amp_y * np.sin(t_demo),
            t_demo / (2 * np.pi)
        ])

        # 5. Smooth interpolation from start_offset to goal_offset
        blend = np.linspace(0, 1, T)[:, None]
        offset = (1 - blend) * start_offset + blend * goal_offset

        # 6. Small local noise on top
        noise = np.random.normal(0, 0.01, size=base.shape)

        trajectories_list.append(base + offset + noise)
    return trajectories_list

def dataset6(nbDemos=1,seed=0,starts=None,ends=None):
    def preprocess_demos(
        demo_dir: str = "arms_demo/D1",
        file_prefix: str = "demo",
        n_steps: int = 100,
        vel_threshold: float = 0.01,
        margin: int = 5,
    ) -> np.ndarray:
        """
        Load and preprocess demonstration files recorded by DemoRecorder.

        Steps:
          1. Load all .npy demo files from demo_dir.
          2. Trim leading/trailing samples where the robot is not moving
             (all joint velocities below vel_threshold).
          3. Resample each demo to exactly n_steps via linear interpolation.
          4. Stack into a matrix of shape (n_demos, n_steps, n_joints).

        Args:
            demo_dir:       Directory containing the .npy demo files.
            file_prefix:    Filename prefix used when saving (default "demo").
            n_steps:        Number of timesteps every demo is resampled to.
            vel_threshold:  A sample is considered "moving" if any joint velocity
                            exceeds this value (rad/s).
            margin:         Extra samples to keep on each side of the active window
                            to avoid clipping the motion onset/offset.

        Returns:
            data: np.ndarray of shape (n_demos, n_steps, n_joints)
                  containing joint positions only.
        """
        # ── 1. Collect files ──────────────────────────────────────────────────────
        paths = [os.path.join(demo_dir, f"demo_00{i}.npy") for i in range(1,11)]

        processed = []

        for path in paths:
            demo = np.load(path, allow_pickle=True).item()   # dict: time, pos, vel, acc
            pos = np.array(demo["pos"])   # (T, n_joints)
            vel = np.array(demo["vel"])   # (T, n_joints)
            t   = np.array(demo["time"])  # (T,)

            # Remove ur5 pos and vel
            pos = pos[:,-9:-2]
            vel = vel[:,-9:-2]

            # ── 2. Trim silent regions ─────────────────────────────────────────
            moving = np.linalg.norm(pos[1:]-pos[:-1],axis=1) > 1e-2  #np.any(np.abs(vel) > vel_threshold, axis=1)  # (T,) bool mask
            if not moving.any():
                print(f"  WARNING: {os.path.basename(path)} has no motion above "
                      f"threshold={vel_threshold}. Skipping.")
                continue
            first = max(moving.argmax() - margin, 0)
            last  = min(len(moving) - 1 - moving[::-1].argmax() + margin,len(moving) - 1)

            pos_trim = pos[first:last + 1]   # (T', n_joints)
            t_trim   = t[first:last + 1]     # (T',)
            if len(t_trim) < 2:
                print(f"  WARNING: {os.path.basename(path)} too short after trimming. Skipping.")
                continue

            # ── 3. Resample to n_steps via linear interpolation ────────────────
            #t_norm   = (t_trim - t_trim[0]) / (t_trim[-1] - t_trim[0])  # [0, 1]
            #t_target = np.linspace(0.0, 1.0, n_steps)
    #
            #interpolator = interp1d(t_norm, pos_trim, axis=0, kind="linear")
            #pos_resampled = interpolator(t_target)  # (n_steps, n_joints)
    #
            #processed.append(pos_resampled)
            #print(f"  {os.path.basename(path)}: {len(t)} → trimmed {len(t_trim)} "
            #      f"→ resampled {n_steps} steps, {pos_resampled.shape[1]} joints.")
            processed.append(pos_trim)

        if not processed:
            raise RuntimeError("No valid demos after preprocessing.")

        # ── 4. Stack into (n_demos, n_steps, n_joints) ────────────────────────────
        trimmed = []
        target_len = min(d.shape[0] for d in processed)
        for d in processed:
            n = d.shape[0]
            # pick target_len indices spread uniformly over [0, n-1]
            idx = np.round(np.linspace(0, n - 1, target_len)).astype(int)
            trimmed.append(d[idx])
        data = np.stack(trimmed, axis=0)
        print(f"\nFinal matrix shape: {data.shape}  "
              f"(n_demos={data.shape[0]}, n_steps={data.shape[1]}, n_joints={data.shape[2]})")
        return data
    
    def reconstruct_demo(mean_traj, W, z):
        D, T = mean_traj.shape
        mean_flat = mean_traj.reshape(-1)
        x = mean_flat + z @ W
        return x.reshape(D, T)
    def sample_latent(Z):
        z_mean = Z.mean(axis=0)
        z_cov = np.cov(Z.T)
        return np.random.multivariate_normal(z_mean, z_cov)
    def sample_demo(mean_traj, W, Z):
        z = sample_latent(Z)
        return reconstruct_demo(mean_traj, W, z)
    def sample_demos(mean_traj, W, Z, n_samples=5):
        return [sample_demo(mean_traj, W, Z) for _ in range(n_samples)] #np.stack([sample_demo(mean_traj, W, Z) for _ in range(n_samples)])
    def sample_demos_constrained(mean_traj, W, x_start, x_end, n_samples=5, eps=1e-6):
        T, D = mean_traj.shape
        mu = mean_traj.reshape(-1)
        # trajectory covariance induced by latent model
        Sigma_x = W.T @ W + eps * np.eye(T * D)
        # observation matrix (start + end)
        A = np.zeros((6, T * D))
        A[:3, :3] = np.eye(3)
        A[3:, -3:] = np.eye(3)
        y = np.hstack([x_start, x_end])
        # conditioning
        Sigma_y = A @ Sigma_x @ A.T + eps * np.eye(6)
        K = Sigma_x @ A.T @ np.linalg.inv(Sigma_y)
        mu_cond = mu + K @ (y - A @ mu)
        Sigma_cond = Sigma_x - K @ A @ Sigma_x
        # sample trajectories in flat space
        L = np.linalg.cholesky(Sigma_cond + eps * np.eye(T * D))
        samples = []
        for _ in range(n_samples):
            x = mu_cond + L @ np.random.randn(T * D)
            samples.append(x.reshape(T, D))
        return samples #np.stack(samples)
    def fit_latent_function(demos, latent_dim=20):
        """
        demos: (n_demos, n_joints, n_steps)
        returns:
            mean_traj: (n_steps, n_joints)   ← transposed back for downstream functions
            W: (latent_dim, n_joints*n_steps)
            Z: (n_demos, latent_dim)
        """
        n_demos, D, T = demos.shape

        X = demos.reshape(n_demos, D * T)

        mean_traj = X.mean(axis=0, keepdims=True)
        X_centered = X - mean_traj

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        W = Vt[:latent_dim]
        Z = X_centered @ W.T

        return mean_traj.reshape(D, T).T, W, Z
    def dataset_from_real(demos: np.ndarray, nbDemos=1, starts=None, ends=None):
        """
        Args:
            demos: np.ndarray of shape (n_demos, n_steps, n_joints)
                   as returned by preprocess_demos()
        """
        # dataset3 expects (n_demos, n_features, n_steps) → transpose
        demos_t = demos.transpose(0, 2, 1)   # (n_demos, n_joints, n_steps)

        mean_traj, W, Z = fit_latent_function(demos_t)

        if starts is None:
            samples = sample_demos(mean_traj, W, Z, n_samples=nbDemos)
        else:
            samples = sample_demos_constrained(
                mean_traj=mean_traj,
                W=W,
                x_start=starts.squeeze(),
                x_end=ends.squeeze(),
                n_samples=nbDemos
            )

        return samples
    
    demos = preprocess_demos(
        demo_dir=demo_dir,
        n_steps=100,          # target timeline length
        vel_threshold=0.01,   # rad/s — tune to your robot's noise floor
        margin=5,             # samples to keep around motion onset/offset
    )
    samples = dataset_from_real(demos, nbDemos)

    if seed<10:
        samples = []
        for i in range(nbDemos):
            samples.append(demos[seed-i])
    return samples

def dataset7(nbDemos=1,seed=0,starts=None,ends=None):
    
    def preprocess_demos(
        demo_dir: str = "arms_demo/D1",
        file_prefix: str = "demo",
        n_steps: int = 100,
        vel_threshold: float = 0.01,
        margin: int = 5,
    ) -> np.ndarray:
        """
        Load and preprocess demonstration files recorded by DemoRecorder.

        Steps:
          1. Load all .npy demo files from demo_dir.
          2. Trim leading/trailing samples where the robot is not moving
             (all joint velocities below vel_threshold).
          3. Resample each demo to exactly n_steps via linear interpolation.
          4. Stack into a matrix of shape (n_demos, n_steps, n_joints).

        Args:
            demo_dir:       Directory containing the .npy demo files.
            file_prefix:    Filename prefix used when saving (default "demo").
            n_steps:        Number of timesteps every demo is resampled to.
            vel_threshold:  A sample is considered "moving" if any joint velocity
                            exceeds this value (rad/s).
            margin:         Extra samples to keep on each side of the active window
                            to avoid clipping the motion onset/offset.

        Returns:
            data: np.ndarray of shape (n_demos, n_steps, n_joints)
                  containing joint positions only.
        """
        # ── 1. Collect files ──────────────────────────────────────────────────────
        paths = [os.path.join(demo_dir, f"demo_00{i}.npy") for i in range(1,11)]

        processed = []

        for path in paths:
            demo = np.load(path, allow_pickle=True).item()   # dict: time, pos, vel, acc
            pos = np.array(demo["pos"])   # (T, n_joints)
            vel = np.array(demo["vel"])   # (T, n_joints)
            t   = np.array(demo["time"])  # (T,)

            # Remove ur5 pos and vel
            pos = pos[:,-9:-2]
            vel = vel[:,-9:-2]

            # ── 2. Trim silent regions ─────────────────────────────────────────
            moving = np.linalg.norm(pos[1:]-pos[:-1],axis=1) > 1e-2  #np.any(np.abs(vel) > vel_threshold, axis=1)  # (T,) bool mask
            if not moving.any():
                print(f"  WARNING: {os.path.basename(path)} has no motion above "
                      f"threshold={vel_threshold}. Skipping.")
                continue
            first = max(moving.argmax() - margin, 0)
            last  = min(len(moving) - 1 - moving[::-1].argmax() + margin,len(moving) - 1)

            pos_trim = pos[first:last + 1]   # (T', n_joints)
            t_trim   = t[first:last + 1]     # (T',)
            if len(t_trim) < 2:
                print(f"  WARNING: {os.path.basename(path)} too short after trimming. Skipping.")
                continue

            # ── 3. Resample to n_steps via linear interpolation ────────────────
            #t_norm   = (t_trim - t_trim[0]) / (t_trim[-1] - t_trim[0])  # [0, 1]
            #t_target = np.linspace(0.0, 1.0, n_steps)
    #
            #interpolator = interp1d(t_norm, pos_trim, axis=0, kind="linear")
            #pos_resampled = interpolator(t_target)  # (n_steps, n_joints)
    #
            #processed.append(pos_resampled)
            #print(f"  {os.path.basename(path)}: {len(t)} → trimmed {len(t_trim)} "
            #      f"→ resampled {n_steps} steps, {pos_resampled.shape[1]} joints.")
            processed.append(pos_trim)

        if not processed:
            raise RuntimeError("No valid demos after preprocessing.")

        # ── 4. Stack into (n_demos, n_steps, n_joints) ────────────────────────────
        trimmed = []
        target_len = min(d.shape[0] for d in processed)
        print(target_len)
        for d in processed:
            n = d.shape[0]
            # pick target_len indices spread uniformly over [0, n-1]
            idx = np.round(np.linspace(0, n - 1, target_len)).astype(int)
            trimmed.append(d[idx])
        data = np.stack(trimmed, axis=0)
        print(f"\nFinal matrix shape: {data.shape}  "
              f"(n_demos={data.shape[0]}, n_steps={data.shape[1]}, n_joints={data.shape[2]})")
        return data
    
    def reconstruct_demo(mean_traj, W, z):
        D, T = mean_traj.shape
        mean_flat = mean_traj.reshape(-1)
        x = mean_flat + z @ W
        return x.reshape(D, T)
    def sample_latent(Z):
        z_mean = Z.mean(axis=0)
        z_cov = np.cov(Z.T)
        return np.random.multivariate_normal(z_mean, z_cov)
    def sample_demo(mean_traj, W, Z):
        z = sample_latent(Z)
        return reconstruct_demo(mean_traj, W, z)
    def sample_demos(mean_traj, W, Z, n_samples=5):
        return [sample_demo(mean_traj, W, Z) for _ in range(n_samples)] #np.stack([sample_demo(mean_traj, W, Z) for _ in range(n_samples)])
    def sample_demos_constrained(mean_traj, W, x_start, x_end, n_samples=5, eps=1e-6):
        T, D = mean_traj.shape
        mu = mean_traj.reshape(-1)
        # trajectory covariance induced by latent model
        Sigma_x = W.T @ W + eps * np.eye(T * D)
        # observation matrix (start + end)
        A = np.zeros((6, T * D))
        A[:3, :3] = np.eye(3)
        A[3:, -3:] = np.eye(3)
        y = np.hstack([x_start, x_end])
        # conditioning
        Sigma_y = A @ Sigma_x @ A.T + eps * np.eye(6)
        K = Sigma_x @ A.T @ np.linalg.inv(Sigma_y)
        mu_cond = mu + K @ (y - A @ mu)
        Sigma_cond = Sigma_x - K @ A @ Sigma_x
        # sample trajectories in flat space
        L = np.linalg.cholesky(Sigma_cond + eps * np.eye(T * D))
        samples = []
        for _ in range(n_samples):
            x = mu_cond + L @ np.random.randn(T * D)
            samples.append(x.reshape(T, D))
        return samples #np.stack(samples)
    def fit_latent_function(demos, latent_dim=20):
        """
        demos: (n_demos, n_joints, n_steps)
        returns:
            mean_traj: (n_steps, n_joints)   ← transposed back for downstream functions
            W: (latent_dim, n_joints*n_steps)
            Z: (n_demos, latent_dim)
        """
        n_demos, D, T = demos.shape

        X = demos.reshape(n_demos, D * T)

        mean_traj = X.mean(axis=0, keepdims=True)
        X_centered = X - mean_traj

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        W = Vt[:latent_dim]
        Z = X_centered @ W.T

        return mean_traj.reshape(D, T).T, W, Z
    def dataset_from_real(demos: np.ndarray, nbDemos=1, starts=None, ends=None):
        """
        Args:
            demos: np.ndarray of shape (n_demos, n_steps, n_joints)
                   as returned by preprocess_demos()
        """
        # dataset3 expects (n_demos, n_features, n_steps) → transpose
        demos_t = demos.transpose(0, 2, 1)   # (n_demos, n_joints, n_steps)

        mean_traj, W, Z = fit_latent_function(demos_t)

        if starts is None:
            samples = sample_demos(mean_traj, W, Z, n_samples=nbDemos)
        else:
            samples = sample_demos_constrained(
                mean_traj=mean_traj,
                W=W,
                x_start=starts.squeeze(),
                x_end=ends.squeeze(),
                n_samples=nbDemos
            )

        return samples
    
    demos = preprocess_demos(
        demo_dir=demo_dir,
        n_steps=100,          # target timeline length
        vel_threshold=0.01,   # rad/s — tune to your robot's noise floor
        margin=5,             # samples to keep around motion onset/offset
    )
    samples = dataset_from_real(demos, nbDemos)
    samples_pee = [get_ee_trajectory(urdf_path, samples[i])[0] for i in range(nbDemos)]
    if seed<10:
        samples_pee = []
        for i in range(nbDemos):
            samples_pee.append(get_ee_trajectory(urdf_path, demos[seed-i])[0])
    return samples_pee


def create_dataset(traj_label, nbDemos=1, seed=0,starts=None,ends=None):
    # Create demo dataset
    if traj_label == 'dataset5':
        return dataset5(nbDemos=nbDemos, seed=seed,starts=starts,ends=ends)
    elif traj_label == 'dataset4':
        return dataset4(nbDemos=nbDemos, seed=seed,starts=starts,ends=ends) 
    elif traj_label == "dataset1":
       return dataset1(nbDemos=nbDemos, seed=seed,starts=starts,ends=ends) 
    elif traj_label == "dataset2":
        return dataset2(nbDemos=nbDemos, seed=seed,starts=starts,ends=ends)
    elif traj_label == "dataset3":
        return dataset3(nbDemos=nbDemos, seed=seed,starts=starts,ends=ends)
    elif traj_label == "dataset6":
        return dataset6(nbDemos=nbDemos, seed=seed,starts=starts,ends=ends)
    elif traj_label == "dataset7":
        return dataset7(nbDemos=nbDemos, seed=seed,starts=starts,ends=ends)
    
trajectories_list = create_dataset(traj_label, nbDemos)
trajectories = np.array(trajectories_list)

# Plot demo
fig, axs = plt.subplots(2, nbStates, figsize=(5*nbStates,3 * 2))
fig.suptitle('Demonstrations')
plt.subplots_adjust(hspace=0.5)
for feature_idx in range(nbStates):
    axs[0][feature_idx].set_title(POSITION_FEATURES[feature_idx])
    for traj_idx in range(len(trajectories)):
        axs[0][feature_idx].plot(trajectories[traj_idx][:,feature_idx])
plt.show()



# --- time axis ---
tau = 1.0
times = torch.linspace(0, tau, T)

# --- position-only demo tensor: [nbDemos, T, nbStates] ---
demos_pos = torch.tensor(
    np.stack([t for t in trajectories_list], axis=0),
    dtype=torch.float32
)
demos_pos = torch.tensor(trajectories,dtype=torch.float32)

# --- build ProDMP (requires additional args vs ProMP) ---
prodmp = MPFactory.init_mp(**deepcopy(mp_config))

# --- learn weights ---
times_batched = times.unsqueeze(0).expand(nbDemos, -1)
params_dict = prodmp.learn_mp_params_from_trajs(
    times=times_batched,
    trajs=demos_pos
)

# --- mean and covariance ---
params_mean = params_dict['params'].mean(dim=0)
params_cov  = torch.cov(params_dict['params'].T)
params_L    = cholesky(
    params_cov + 1e-6 * torch.eye(params_cov.shape[0])
)

###mean_init_pos = demos_pos[:, 0, :].mean(dim=0)
###mean_init_vel = torch.zeros(nbStates)
###
###prodmp.update_inputs(
###    times=times,
###    params=params_mean,
###    params_L=params_L,
###    init_time=torch.tensor(0.0),
###    init_pos=mean_init_pos,
###    init_vel=mean_init_vel,
###)

from mp_pytorch.mp import ProDMP
from faprodmp.utils import cholesky
def position_condition(fa_prodmp: ProDMP, 
                    current_trajectory: torch.Tensor,
                    time_idx: int, 
                    measured_positions: torch.Tensor,
                    position_threshold: float,
                    position_sum_threshold: float = 0.05,
                    reg: float = 1e-20
                    ) -> ProDMP:
    """conditions a FA-ProDMP based on measured positions

    Args:
        fa_prodmp (ProDMP): the base FA-ProDMP
        current_trajectory (torch.Tensor): current planned trajectory
        time_idx (int): the time index at which the conditioning takes place
        measured_positions (torch.Tensor): the measured positions, scaled in the FA-ProDMP position scale
        position_threshold (float): the threshold at which to condition, scaled in the FA-ProDMP position scale
        position_sum_threshold (float, optional): the ratio of the threshold the conditioning position should reach. Defaults to 0.5.
        reg (float, optional): regularization term for numerical stability. Defaults to 1e-20.

    Returns:
        ProDMP: the conditioned FA-ProDMP
    """
    # initialize conditioned FA-ProDMP
    conditioned_fa_prodmp = deepcopy(fa_prodmp)
    times = conditioned_fa_prodmp.times

    num_dof_pos = measured_positions.shape[0]
    num_dof_force = conditioned_fa_prodmp.num_dof - num_dof_pos
    weight_mean_old = conditioned_fa_prodmp.params.unsqueeze(1)
    weight_cov_old = conditioned_fa_prodmp.params_L @ conditioned_fa_prodmp.params_L.T

    # initialize conditioning vector
    conditioning = torch.cat((
        measured_positions,
        ), dim = 0)

    # get prediction
    pos_traj_old = conditioned_fa_prodmp.get_traj_pos() #
    prediction = pos_traj_old[time_idx].unsqueeze(1) #

    # get noise needed for numerical stability
    noise = torch.eye(conditioning.shape[0])
    noise[:num_dof_pos, :num_dof_pos] *= reg

    abs_position_differences = torch.abs(conditioning[:num_dof_pos].squeeze() - current_trajectory[time_idx][:num_dof_pos])
    sorted_indices = torch.argsort(abs_position_differences, descending=True)
    position_indices = [idx for idx in sorted_indices.tolist() 
                    if abs_position_differences[idx] >= position_threshold]

    conditioned_fa_prodmp.compute_intermediate_terms_multi_dof()
    pos_H = conditioned_fa_prodmp.pos_H_multi * conditioned_fa_prodmp.weights_goal_scale.repeat(conditioned_fa_prodmp.num_dof)
    resolved_idx = time_idx % times.shape[0]
    relevant_columns = list(range(resolved_idx, pos_H.shape[0], times.shape[0]))

    pos_H_t = torch.zeros_like(pos_H[relevant_columns])
    pos_H_t[position_indices] = pos_H[relevant_columns][position_indices]

    kalman_gain = weight_cov_old @ pos_H_t.T @ torch.linalg.inv(pos_H_t @ weight_cov_old @ pos_H_t.T + noise)

    weight_mean_new = weight_mean_old + kalman_gain @ (conditioning - prediction)
    weight_cov_new = weight_cov_old - kalman_gain @ pos_H_t @ weight_cov_old

    params_new = weight_mean_new.squeeze()
    params_L_new = cholesky(weight_cov_new)

    conditioned_fa_prodmp.update_inputs(
        times=times,
        params=params_new,
        params_L=params_L_new,
        init_time=conditioned_fa_prodmp.init_time,
        init_pos=conditioned_fa_prodmp.init_pos,
        init_vel=conditioned_fa_prodmp.init_vel
    )

    return conditioned_fa_prodmp

def test(query_translations, frame_idx, name, prodmp, original_traj=None, shift_init=1,shift_end=1, plot_flag=1):
    mean_init_vel = torch.zeros(nbStates)

    if traj_label == 'dataset1' or traj_label == 'dataset2':
        ends = np.array([0.4, 0.2, 0.4])
        starts = np.array([-0.6, -0.2, -0.6])
    elif traj_label == 'dataset3':
        starts = np.array([-0.8, -1, -0.8])
        ends = np.array([-0.25, 1, 0.25])
    elif traj_label == 'dataset4' or  traj_label == 'dataset5':
        starts=np.array([1, 0, 0])
        ends=np.array([1, 0, 1])

    #mean_traj_orig = prodmp.get_traj_pos()   # [T, nbStates]
    #std_orig       = prodmp.get_traj_pos_std()

    # --- shifted: only init_pos changes, works natively in ProDMP ---
    query_translations = [
        torch.tensor(q, dtype=torch.float32) if not isinstance(q, torch.Tensor) else q
        for q in query_translations
    ]
    prodmp_shifted_init = deepcopy(prodmp)
    if shift_init:
        prodmp_shifted_init.update_inputs(
            times=times,
            params=params_mean,
            params_L=params_L,
            init_time=torch.tensor(0.0),
            init_pos=query_translations[0],
            init_vel=mean_init_vel,
        )
    if shift_end:
        prodmp_shifted = deepcopy(prodmp_shifted_init)
        prodmp_shifted = position_condition(fa_prodmp=prodmp_shifted_init,
                                        current_trajectory=prodmp_shifted_init.get_traj_pos(),
                                        time_idx=frame_idx[-1],
                                        measured_positions=query_translations[-1][None,:].T,
                                        position_threshold=0.001,
                                        #reg=1e-6
                                    )
    mean_traj_shifted = prodmp_shifted.get_traj_pos()
    std_shifted       = prodmp_shifted.get_traj_pos_std()

    if plot_flag:
        # --- plot ---
        fig, axs = plt.subplots(1, nbStates, figsize=(5 * nbStates, 4))
        fig.suptitle(name)
        times_np = times.numpy()

        for feature_idx, feature_name in enumerate(POSITION_FEATURES):
            ax = axs[feature_idx]
            ax.set_title(feature_name)

            # shifted
            mean_shifted_np = mean_traj_shifted[:, feature_idx].detach().numpy()
            std_shifted_np  = std_shifted[:, feature_idx].detach().numpy()
     
            ax.fill_between(times_np,
                            mean_shifted_np - STD_SCALE * std_shifted_np,
                            mean_shifted_np + STD_SCALE * std_shifted_np,
                            color='orange', alpha=0.3)
            ax.plot(times_np, mean_shifted_np, color='orange', label='shifted')

            if original_traj is not None:
                ax.plot(times_np, original_traj[0][:,feature_idx], color='green', label='demo')

            # mark new start
            for (MEASUREMENT_PHASE_IDX_i, query_translations_i) in zip(frame_indices, query_translations):
                ax.plot(times_np[MEASUREMENT_PHASE_IDX_i], query_translations_i[feature_idx].item(),
                        'x', ms=10)
            ax.legend()

            if traj_label != 'dataset6' and traj_label != 'dataset7':
                if traj_label == 'dataset1' or traj_label == 'dataset2':
                    ends = np.array([0.4, 0.2, 0.4])
                    starts = np.array([-0.6, -0.2, -0.6])
                elif traj_label == 'dataset3':
                    starts = np.array([-0.8, -1, -0.8])
                    ends = np.array([-0.25, 1, 0.25])
                elif traj_label == 'dataset4' or  traj_label == 'dataset5':
                    starts=np.array([1, 0, 0])
                    ends=np.array([1, 0, 1])

                ax.scatter(times[0], starts[feature_idx],
                    s=50,
                    label=f"mean starts demo",
                    c='k',)
                ax.scatter(times[-1], ends[feature_idx],
                    s=50,
                    label=f"mean ends demo",
                    c='k',)
                ax.scatter(times[-1], ends[feature_idx]+0.5,
                    s=50,
                    label=f"mean+scale ends demo",
                    marker='+',
                    c='k',)
                ax.scatter(times[-1], ends[feature_idx]-0.5,
                    s=50,
                    label=f"mean+scale ends demo",
                    marker='+',
                    c='k',)

                ax.set_ylim([-1.5,1.5])

        plt.tight_layout()

        plt.savefig(f'{fig_dir}/{name}.png')
        plt.close()

    print(f"Distance to start: {np.linalg.norm(query_translations[0] - mean_traj_shifted[0]):.4f}")
    print(f"Distance to end: {np.linalg.norm(query_translations[-1] - mean_traj_shifted[-1]):.4f}")

    return mean_traj_shifted

###############################################################################
if traj_label == 'dataset6':
    fig, axs = plt.subplots(2, 5)
    trajectories_demo = create_dataset(traj_label, nbDemos=10)
    pos_ref = [get_ee_trajectory(urdf_path, trajectories_demo[i])[0] for i in range(10)]
    pos_prodmp = [get_ee_trajectory(urdf_path, test(np.array([trajectories_demo[i][0], trajectories_demo[i][-1]]), frame_indices, 'traj_new_s', prodmp, original_traj=trajectories_demo[i], plot_flag=0))[0] for i in range(10)]
    for ii in range(10):
        for jj in range(10):
            axs[ii//5,ii%5].plot(pos_ref[jj][:,0],pos_ref[jj][:,1], alpha=0.3)
        axs[ii//5,ii%5].plot(pos_prodmp[ii][:,0],pos_prodmp[ii][:,1], label='TPGMM')
        axs[ii//5,ii%5].scatter(pos_prodmp[ii][frame_indices,0],pos_prodmp[ii][frame_indices,1], s=50)
    plt.legend()
    plt.show()

if traj_label == 'dataset7':
    fig, axs = plt.subplots(2, 5)
    pos_ref = create_dataset(traj_label, nbDemos=10)
    pos_prodmp = [test(np.array([pos_ref[i][0], pos_ref[i][-1]]), frame_indices, 'traj_new_s', prodmp, original_traj=pos_ref[i], plot_flag=0) for i in range(10)]
    for ii in range(10):
        for jj in range(10):
            axs[ii//5,ii%5].plot(pos_ref[jj][:,0],pos_ref[jj][:,1], alpha=0.3)
        axs[ii//5,ii%5].plot(pos_prodmp[ii][:,0],pos_prodmp[ii][:,1], label='TPGMM')
        axs[ii//5,ii%5].scatter(pos_prodmp[ii][frame_indices,0],pos_prodmp[ii][frame_indices,1], s=50)
    plt.legend()
    plt.show()

###############################################################################
print('----- Test with new y0 and g (same generators) (ProDMP) -----')
frame_indices = [0, -1]

trajectory = np.array(create_dataset(traj_label))
query_translations = np.array([
    trajectory[0][0],
    trajectory[0][-1],
])

test(query_translations, frame_indices, 'traj_new1', prodmp, original_traj=trajectory)

###############################################################################
print('----- Test with new y0 and g (same generators) (ProDMP) -----')
frame_indices = [0, -1]

trajectory = np.array(create_dataset(traj_label,seed=10))
query_translations = np.array([
    trajectory[0][0],
    trajectory[0][-1],
])

test(query_translations, frame_indices, 'traj_new2', prodmp, original_traj=trajectory)

###############################################################################
print('----- Test with new y0 and g (same generators) (ProDMP) -----')
frame_indices = [0, -1]

trajectory = np.array(create_dataset(traj_label,seed=11))
query_translations = np.array([
    trajectory[0][0],
    trajectory[0][-1],
])

test(query_translations, frame_indices, 'traj_new3', prodmp, original_traj=trajectory)



###############################################################################

if traj_label == 'dataset1' or traj_label == 'dataset2':
    traj_mean = dataset2(starts=np.array([[-0.6, -0.2, -0.6]]),ends=np.array([[0.4, 0.2, 0.4]])) #o 2
elif traj_label == 'dataset3':
    traj_mean = dataset3(starts=np.array([-0.8, -1, -0.8]),ends=np.array([-0.25, 1, 0.25]))
elif traj_label == 'dataset4':
    traj_mean = dataset4(starts=np.array([1, 0, 0]),ends=np.array([1, 0, 1]))
elif traj_label == 'dataset5':
    traj_mean = dataset5(starts=np.array([-0., 0, -0.]),ends=np.array([0, 0, 0])) #*0 the starts and ends in trajectory
elif traj_label == 'dataset6':
    traj_mean = dataset6(seed=0)
elif traj_label == 'dataset7':
    traj_mean = dataset7(seed=0)

###############################################################################
print('----- Test with shifted initial position 0.05 -----')

query_translations = np.array([
    traj_mean[0][0] + 0.05*np.ones(nbStates),
    traj_mean[0][-1],
])

trajectory = np.array(create_dataset(traj_label,starts=np.array([query_translations[0]]),ends=np.array([query_translations[1]])))

test(query_translations, frame_indices, 'y0_1', prodmp, original_traj=trajectory)

###############################################################################
print('----- Test with shifted initial position 0.1 -----')

query_translations = np.array([
    traj_mean[0][0] + 0.1*np.ones(nbStates),
    traj_mean[0][-1],
])

trajectory = np.array(create_dataset(traj_label,starts=np.array([query_translations[0]]),ends=np.array([query_translations[1]])))

test(query_translations, frame_indices, 'y0_2', prodmp, original_traj=trajectory)

###############################################################################
print('----- Test with shifted initial position 0.5 -----')

query_translations = np.array([
    traj_mean[0][0] + 0.5*np.ones(nbStates),
    traj_mean[0][-1],
])

trajectory = np.array(create_dataset(traj_label,starts=np.array([query_translations[0]]),ends=np.array([query_translations[1]])))

test(query_translations, frame_indices, 'y0_3', prodmp, original_traj=trajectory)

###############################################################################
print('----- Test with shifted goal position 0.05 -----')
query_translations = np.array([
    traj_mean[0][0],
    traj_mean[0][-1] + 0.05*np.ones(nbStates),
])

trajectory = np.array(create_dataset(traj_label,starts=np.array([query_translations[0]]),ends=np.array([query_translations[1]])))

test(query_translations, frame_indices, 'g_1', prodmp, original_traj=trajectory)

###############################################################################
print('----- Test with shifted goal position +0.1 (ProDMP) -----')
query_translations = np.array([
    traj_mean[0][0],
    traj_mean[0][-1] + 0.1*np.ones(nbStates),
])

trajectory = np.array(create_dataset(traj_label,starts=np.array([query_translations[0]]),ends=np.array([query_translations[1]])))

test(query_translations, frame_indices, 'g_2', prodmp, original_traj=trajectory)#,shift_init=0)

###############################################################################
print('----- Test with shifted goal position +0.5 (ProDMP) -----')

query_translations = np.array([
    traj_mean[0][0],
    traj_mean[0][-1] + 0.5*np.ones(nbStates),
])

trajectory = np.array(create_dataset(traj_label,starts=np.array([query_translations[0]]),ends=np.array([query_translations[1]])))

test(query_translations, frame_indices, 'g_3', prodmp, original_traj=trajectory)#,shift_init=0)

###############################################################################

# use a scaled identity as prior — much more stable
params_cov = torch.cov(params_dict['params'].T)

# blend with scaled identity to ensure well-conditioned covariance
cov_scale = params_cov.diag().mean()  # scale to match data magnitude
params_cov_stable = params_cov + cov_scale * torch.eye(params_cov.shape[0]) *1e-6

params_L = cholesky(
    params_cov_stable + 1e-6 * torch.eye(params_cov.shape[0])
)

print('----- Test with shifted initial position 0.5 -----')

query_translations = np.array([
    traj_mean[0][0],
    traj_mean[0][-1]+ 0.5*np.ones(nbStates),
])

trajectory = np.array(create_dataset(traj_label,starts=np.array([query_translations[0]]),ends=np.array([query_translations[1]])))

test(query_translations, frame_indices, 'g_3_var', prodmp, original_traj=trajectory)
