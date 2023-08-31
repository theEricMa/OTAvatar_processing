"""This script defines the face reconstruction model for Deep3DFaceRecon_pytorch
"""

import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .bfm import ParametricFaceModel

from util import util
from scipy.io import savemat

class FaceReconModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        # net structure and parameters
        parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
        parser.add_argument('--init_path', type=str, default='checkpoints/init_model/resnet50-0676ba61.pth')
        parser.add_argument('--use_last_fc', type=util.str2bool, nargs='?', const=True, default=False, help='zero initialize the last fc')
        parser.add_argument('--bfm_folder', type=str, default='BFM')
        parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

        # renderer parameters
        parser.add_argument('--focal', type=float, default=1015.)
        parser.add_argument('--center', type=float, default=112.)
        parser.add_argument('--camera_d', type=float, default=10.)
        parser.add_argument('--z_near', type=float, default=5.)
        parser.add_argument('--z_far', type=float, default=15.)
        parser.add_argument('--use_opengl', type=util.str2bool, nargs='?', const=True, default=True, help='use opengl context or not')

        if is_train:
            raise NotImplementedError('Training is not supported yet.')

        opt, _ = parser.parse_known_args()
        parser.set_defaults(
                focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
            )
        if is_train:
            parser.set_defaults(
                use_crop_face=True, use_predef_M=False
            )
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        
        self.visual_names = ['output_vis']
        self.model_names = ['net_recon']

        self.net_recon = networks.define_net_recon(
            net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path
        )

        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=self.isTrain, default_name=opt.bfm_model
        )
        

        if self.isTrain:
            raise NotImplementedError('Training is not supported yet.')
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['imgs'].to(self.device) 
        self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
        self.gt_lm = input['lms'].to(self.device)  if 'lms' in input else None
        self.trans_m = input['M'].to(self.device) if 'M' in input else None
        self.image_paths = input['im_paths'] if 'im_paths' in input else None

    def forward(self):
        self.net_recon.to(self.device)
        output_coeff = self.net_recon(self.input_img)
        self.facemodel.to(self.device)
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = self.facemodel.compute_for_render(output_coeff)
        self.pred_coeffs_dict = self.facemodel.split_coeff(output_coeff)

    def compute_losses(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        raise NotImplementedError('Training is not supported yet.')

    def optimize_parameters(self, isTrain=True):
        self.forward()               
        self.compute_losses()
        """Update network weights; it will be called in every training iteration."""
        if isTrain:
            self.optimizer.zero_grad()  
            self.loss_all.backward()         
            self.optimizer.step()        

    def compute_visuals(self):
        raise NotImplementedError('Visualization is not supported yet.')

    def save_mesh(self, name):
        raise NotImplementedError('Saving mesh is not supported yet.')

    def save_coeff(self,name):

        pred_coeffs = {key:self.pred_coeffs_dict[key].cpu().numpy() for key in self.pred_coeffs_dict}
        pred_lm = self.pred_lm.cpu().numpy()
        pred_lm = np.stack([pred_lm[:,:,0],self.input_img.shape[2]-1-pred_lm[:,:,1]],axis=2) # transfer to image coordinate
        pred_coeffs['lm68'] = pred_lm
        savemat(name,pred_coeffs)



