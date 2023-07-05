from torch.backends import cudnn
from utils import recon_ops
from FistaNetModel import *
from utils import *
import time
from datetime import datetime
from pytools import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True

root_dir = '/home/10T/DreamNet/Data'

model_name = 'FistaNet_FBP_N4_V72'

views = 576
sparse_rate = 8

results_save_dir = './runs/' + model_name + '/test/'
make_dirs(results_save_dir)

epoch = 100
model_dir = './runs/' + model_name + '/checkpoints/model_at_epoch_' + str(epoch).rjust(3, '0') + '.dat'
checkpoint = torch.load(model_dir)
angles = np.linspace(0, 2*np.pi, views, endpoint=False)
sparse_angles = angles[::sparse_rate]
op_example = recon_ops(angles=sparse_angles)

model = FISTANet(op_example, 7)

model = load_model(model, checkpoint).cuda()
model.eval()

# input1 = torch.nn.Parameter(torch.FloatTensor(1, 1, 72, 736).fill_(0.1)).cuda()
# input2 = torch.nn.Parameter(torch.FloatTensor(1, 1, 512, 512).fill_(0.1)).cuda()
# flops, params = profile(model, inputs=(input2, input1))
# print(round(flops / 1000 ** 3, 2))
# print(round(params / 1000 ** 2, 2))

test_cases = ['L067']

for case in test_cases:

    hdct_path = root_dir + '/AAPM/' + case + '/full_1mm/'
    hdct_vol = read_dicom_all(hdct_path, 20, 24)
    hdct_vol = hdct_vol - 1024

    pred_vol = np.zeros(np.shape(hdct_vol), dtype=np.float32)

    ldct_vol = read_raw_data_all(root_dir + '/sAAPMImg/' + case + '/sparse_ct_v' + str(views//sparse_rate) + '_1e6/', w=512, h=512, start_index=0, end_index=-14)
    ldct_vol[ldct_vol < 0] = 0
    ldct_vol = ldct_vol * 1024

    ldproj_vol = read_raw_data_all(root_dir + '/sAAPMProj/' + case + '/noisy_proj_1e6/', w=576, h=736, start_index=0, end_index=-15)
    ldproj_vol[ldproj_vol < 0] = 0

    t1 = time.time()

    for slice in range(0, np.size(ldproj_vol, 0)):

        ldct_slices = ldct_vol[slice, :, :]
        ldproj_slices = ldproj_vol[slice, :, :]

        ldct_slices = ldct_slices[np.newaxis, np.newaxis, ...]
        ldproj_slices = ldproj_slices[np.newaxis, np.newaxis, ...]

        ldCT = torch.FloatTensor(ldct_slices)
        ldProj = torch.FloatTensor(ldproj_slices)

        ldProj = ldProj[:, :, ::sparse_rate, :]

        ldProj = ldProj.cuda()
        ldCT = ldCT.cuda()

        with torch.no_grad():

            [img_net, _, _] = model(ldCT, ldProj)
            pred_img = np.squeeze(img_net.data.cpu().numpy())
            pred_vol[slice, :, :] = pred_img / 0.02 - 1024


    t2 = time.time()

    print(round(np.size(ldproj_vol, 0) / (t2-t1)))

    pred_vol.astype(np.float32).tofile(results_save_dir + case + '_' + model_name + '_E' + str(epoch) + '.raw')