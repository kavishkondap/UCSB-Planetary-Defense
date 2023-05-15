import numpy as np
import pandas as pd
checkpoint2 = np.array (pd.read_csv ('checkpoint2_gpu.csv')['checkpoint2']).reshape ((62500, 2000))
df = pd.DataFrame (checkpoint2)
df.to_csv ('checkpoint2_gpu_v2.csv')