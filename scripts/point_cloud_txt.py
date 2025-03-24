import glob
import os.path

import numpy as np

# Main script
def main():

    pc_files = glob.glob('../examples/dataset/*/*/pointcloud/*point_cloud.npy')

    for file_name in pc_files:
        pc = np.load(file_name)
        txt_file_name = os.path.join(os.path.dirname(file_name), 'point_cloud.txt')
        np.savetxt(txt_file_name, pc[:, :3], fmt='%.6f', delimiter=',')

if __name__ == "__main__":
    main()