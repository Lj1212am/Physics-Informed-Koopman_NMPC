import scipy.io
import numpy as np
# !pip install pyquaternion
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from scipy.integrate import odeint
import pandas as pd
from scipy.io import loadmat