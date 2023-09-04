import argparse
import numpy as np
from facemodel import Face_3DMM
import cv2
from PIL import Image
import os
import torch
from util import *
from render_3dmm import Render_3DMM
from scipy.io import savemat
from imageio import mimsave


def set_requires_grad(tensor_list):
    for tensor in tensor_list:
        tensor.requires_grad = True

def read_video(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            break
    cap.release()
    return np.stack(frames)

def read_txt(filename, num_frames):
    data = np.loadtxt(filename).astype(np.float32)
    return data.reshape(num_frames, -1, 2)

def run(frames, lms, opt, **args):
    lms = torch.as_tensor(lms).cuda()

    sel_num = lms.shape[0]
    num_frames = lms.shape[0]
    sel_ids = np.arange(0, sel_num)

    id_dim = args.get('id_dim', 100)
    exp_dim = args.get('exp_dim', 79)
    tex_dim = args.get('tex_dim', 100)
    point_dim = args.get('point_dim', 34650)

    model_3dmm = Face_3DMM('./3DMM', id_dim, exp_dim, tex_dim, point_dim, )

    arg_focal = 1600
    arg_landis = 1e5

    cxy = torch.tensor((opt.img_w/2.0, opt.img_h/2.0), dtype=torch.float).cuda()

    frames_result = frames.copy()

    # for focal in range(600, 1200, 100):
    for focal in [1100]: # temporal modification
        id_para = lms.new_zeros((1, id_dim), requires_grad=True)
        exp_para = lms.new_zeros((sel_num, exp_dim), requires_grad=True)
        euler_angle = lms.new_zeros((sel_num, 3), requires_grad=True)
        trans = lms.new_zeros((sel_num, 3), requires_grad=True)
        focal_length = lms.new_zeros(1, requires_grad=False)

        trans.data[:, 2] -= 7
        focal_length.data += focal

        set_requires_grad([id_para, exp_para, euler_angle, trans])

        optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=.1)
        optimizer_frame = torch.optim.Adam(
            [euler_angle, trans], lr=.1)

        for iter in range(2000):
        # for iter in range(1): # temporal modification
            id_para_batch = id_para.expand(sel_num, -1)
            geometry = model_3dmm.get_3dlandmarks(
                id_para_batch, exp_para, euler_angle, trans, focal_length, cxy)
            proj_geo = forward_transform(
                geometry, euler_angle, trans, focal_length, cxy)
            loss_lan = cal_lan_loss(
                proj_geo[:, :, :2], lms[sel_ids].detach())
            loss = loss_lan
            optimizer_frame.zero_grad()
            loss.backward()
            optimizer_frame.step()
           
        for iter in range(2500):
        # for iter in range(1): # temporal modification
            id_para_batch = id_para.expand(sel_num, -1)
            geometry = model_3dmm.get_3dlandmarks(
                id_para_batch, exp_para, euler_angle, trans, focal_length, cxy)
            proj_geo = forward_transform(
                geometry, euler_angle, trans, focal_length, cxy)
            loss_lan = cal_lan_loss(
                proj_geo[:, :, :2], lms[sel_ids].detach())
            loss_regid = torch.mean(id_para*id_para)
            loss_regexp = torch.mean(exp_para*exp_para)
            loss = loss_lan + loss_regid*0.5 + loss_regexp*0.4
            optimizer_idexp.zero_grad()
            optimizer_frame.zero_grad()
            loss.backward()
            optimizer_idexp.step()
            optimizer_frame.step()
            if iter % 100 == 0 and False:
                print(focal, 'poseidexp', iter, loss_lan.item(),
                    loss_regid.item(), loss_regexp.item())
            if iter % 1500 == 0 and iter >= 1500:
                for param_group in optimizer_idexp.param_groups:
                    param_group['lr'] *= 0.2
                for param_group in optimizer_frame.param_groups:
                    param_group['lr'] *= 0.2
    
        print(focal, loss_lan.item(), torch.mean(trans[:, 2]).item())

        if loss_lan.item() < arg_landis:
            arg_landis = loss_lan.item()
            arg_focal = focal

    print('find best focal', arg_focal)

    id_para = lms.new_zeros((1, id_dim), requires_grad=True)
    exp_para = lms.new_zeros((num_frames, exp_dim), requires_grad=True)
    tex_para = lms.new_zeros((1, tex_dim), requires_grad=True)
    euler_angle = lms.new_zeros((num_frames, 3), requires_grad=True)
    trans = lms.new_zeros((num_frames, 3), requires_grad=True)
    light_para = lms.new_zeros((num_frames, 27), requires_grad=True)
    trans.data[:, 2] -= 7
    focal_length = lms.new_zeros(1, requires_grad=True)
    focal_length.data += arg_focal

    set_requires_grad([id_para, exp_para, tex_para, euler_angle, trans, light_para])

    optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=.1)
    optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=1)

    for iter in range(1500):
    # for iter in [1]: # temporal modification
        id_para_batch = id_para.expand(num_frames, -1)
        geometry = model_3dmm.get_3dlandmarks(
            id_para_batch, exp_para, euler_angle, trans, focal_length, cxy)
        proj_geo = forward_transform(
            geometry, euler_angle, trans, focal_length, cxy)
        loss_lan = cal_lan_loss(
            proj_geo[:, :, :2], lms.detach())
        loss = loss_lan
        optimizer_frame.zero_grad()
        loss.backward()
        optimizer_frame.step()
        if iter == 1000:
            for param_group in optimizer_frame.param_groups:
                param_group['lr'] = 0.1

    for param_group in optimizer_frame.param_groups:
        param_group['lr'] = 0.1

    for iter in range(2000):
    # for iter in [1]: # temporal modification
        id_para_batch = id_para.expand(num_frames, -1)
        geometry = model_3dmm.get_3dlandmarks(
            id_para_batch, exp_para, euler_angle, trans, focal_length, cxy)
        proj_geo = forward_transform(
            geometry, euler_angle, trans, focal_length, cxy)
        loss_lan = cal_lan_loss(
            proj_geo[:, :, :2], lms.detach())
        loss_regid = torch.mean(id_para*id_para)
        loss_regexp = torch.mean(exp_para*exp_para)
        loss = loss_lan + loss_regid*0.5 + loss_regexp*0.4
        optimizer_idexp.zero_grad()
        optimizer_frame.zero_grad()
        loss.backward()
        optimizer_idexp.step()
        optimizer_frame.step()
        if iter % 100 == 0 and False:
            print('poseidexp', iter, loss_lan.item(),
                loss_regid.item(), loss_regexp.item())
        if iter % 1000 == 0 and iter >= 1000:
            for param_group in optimizer_idexp.param_groups:
                param_group['lr'] *= 0.2
            for param_group in optimizer_frame.param_groups:
                param_group['lr'] *= 0.2

    print(loss_lan.item(), torch.mean(trans[:, 2]).item())

    batch_size = 50
    device_default = torch.device('cuda:{}'.format(opt.device_id))
    device_render = torch.device('cuda:{}'.format(opt.device_id))

    renderer = Render_3DMM(arg_focal, opt.img_h, opt.img_w, batch_size, device_render)
    renderer.to(device_render)

    sel_ids = np.arange(0, num_frames, int(num_frames/batch_size))[:batch_size]
    sel_imgs = torch.as_tensor(frames[sel_ids]).cuda()
    sel_lms = lms[sel_ids]

    sel_light = light_para.new_zeros((batch_size, 27), requires_grad=True)
    set_requires_grad([sel_light])
    optimizer_tl = torch.optim.Adam([tex_para, sel_light], lr=.1)
    optimizer_id_frame = torch.optim.Adam(
        [euler_angle, trans, exp_para, id_para], lr=.01)

    for iter in range(71):
    # for iter in range(1): # temporal modification
        sel_exp_para, sel_euler, sel_trans = exp_para[sel_ids], euler_angle[sel_ids], trans[sel_ids]
        sel_id_para = id_para.expand(batch_size, -1)
        geometry = model_3dmm.get_3dlandmarks(
            sel_id_para, sel_exp_para, sel_euler, sel_trans, focal_length, cxy)
        proj_geo = forward_transform(
            geometry, sel_euler, sel_trans, focal_length, cxy)
        loss_lan = cal_lan_loss(proj_geo[:, :, :2], sel_lms.detach())
        loss_regid = torch.mean(id_para*id_para)
        loss_regexp = torch.mean(sel_exp_para*sel_exp_para)

        sel_tex_para = tex_para.expand(batch_size, -1)
        sel_texture = model_3dmm.forward_tex(sel_tex_para)
        geometry = model_3dmm.forward_geo(sel_id_para, sel_exp_para)
        rott_geo = forward_rott(geometry, sel_euler, sel_trans)
        render_imgs = renderer(rott_geo.to(device_render),
                            sel_texture.to(device_render),
                            sel_light.to(device_render))
        render_imgs = render_imgs.to(device_default)

        mask = (render_imgs[:, :, :, 3]).detach() > 0.0
        render_proj = sel_imgs.clone()
        render_proj[mask] = render_imgs[mask][..., :3].byte()
        loss_col = cal_col_loss(render_imgs[:, :, :, :3], sel_imgs.float(), mask)
        loss = loss_col + loss_lan*3 + loss_regid*2.0 + loss_regexp*1.0
        if iter > 50:
            loss = loss_col + loss_lan*0.05 + loss_regid*1.0 + loss_regexp*0.8
        optimizer_tl.zero_grad()
        optimizer_id_frame.zero_grad()
        loss.backward()
        optimizer_tl.step()
        optimizer_id_frame.step()
        if iter % 50 == 0 and iter >= 5:
            for param_group in optimizer_id_frame.param_groups:
                param_group['lr'] *= 0.2
            for param_group in optimizer_tl.param_groups:
                param_group['lr'] *= 0.2

    light_mean = torch.mean(sel_light, 0).unsqueeze(0).repeat(num_frames, 1)
    light_para.data = light_mean

    exp_para = exp_para.detach()
    euler_angle = euler_angle.detach()
    trans = trans.detach()
    light_para = light_para.detach()

    for i in range(int((num_frames-1)/batch_size+1)):
        if (i+1)*batch_size > num_frames:
            start_n = num_frames-batch_size
            sel_ids = np.arange(num_frames-batch_size, num_frames)
        else:
            start_n = i*batch_size
            sel_ids = np.arange(i*batch_size, i*batch_size+batch_size)

        sel_lms = torch.as_tensor(lms[sel_ids]).cuda()
        sel_imgs = torch.as_tensor(frames[sel_ids]).cuda()

        sel_exp_para = exp_para.new_zeros(
            (batch_size, exp_dim), requires_grad=True)
        sel_exp_para.data = exp_para[sel_ids].clone()
        sel_euler = euler_angle.new_zeros(
            (batch_size, 3), requires_grad=True)
        sel_euler.data = euler_angle[sel_ids].clone()
        sel_trans = trans.new_zeros((batch_size, 3), requires_grad=True)
        sel_trans.data = trans[sel_ids].clone()
        sel_light = light_para.new_zeros(
            (batch_size, 27), requires_grad=True)
        sel_light.data = light_para[sel_ids].clone()

        set_requires_grad([sel_exp_para, sel_euler, sel_trans, sel_light])

        optimizer_cur_batch = torch.optim.Adam(
            [sel_exp_para, sel_euler, sel_trans, sel_light], lr=0.005)

        sel_id_para = id_para.expand(batch_size, -1).detach()
        sel_tex_para = tex_para.expand(batch_size, -1).detach()

        pre_num = 5
        if i > 0:
            pre_ids = np.arange(
                start_n-pre_num, start_n)

        for iter in range(50):
        # for iter in [1]: # temporal modification
            geometry = model_3dmm.get_3dlandmarks(
                sel_id_para, sel_exp_para, sel_euler, sel_trans, focal_length, cxy)
            proj_geo = forward_transform(
                geometry, sel_euler, sel_trans, focal_length, cxy)
            loss_lan = cal_lan_loss(proj_geo[:, :, :2], sel_lms.detach())
            loss_regexp = torch.mean(sel_exp_para*sel_exp_para)

            sel_geometry = model_3dmm.forward_geo(sel_id_para, sel_exp_para)
            sel_texture = model_3dmm.forward_tex(sel_tex_para)
            geometry = model_3dmm.forward_geo(sel_id_para, sel_exp_para)
            rott_geo = forward_rott(geometry, sel_euler, sel_trans)
            render_imgs = renderer(rott_geo.to(device_render),
                                sel_texture.to(device_render),
                                sel_light.to(device_render))
            render_imgs = render_imgs.to(device_default)

            mask = (render_imgs[:, :, :, 3]).detach() > 0.0

            loss_col = cal_col_loss(
                render_imgs[:, :, :, :3], sel_imgs.float(), mask)

            if i > 0:
                geometry_lap = model_3dmm.forward_geo_sub(id_para.expand(
                    batch_size+pre_num, -1).detach(), torch.cat((exp_para[pre_ids].detach(), sel_exp_para)), model_3dmm.rigid_ids)
                rott_geo_lap = forward_rott(geometry_lap,  torch.cat(
                    (euler_angle[pre_ids].detach(), sel_euler)), torch.cat((trans[pre_ids].detach(), sel_trans)))

                loss_lap = cal_lap_loss([rott_geo_lap.reshape(rott_geo_lap.shape[0], -1).permute(1, 0)],
                                        [1.0])
            else:
                geometry_lap = model_3dmm.forward_geo_sub(
                    id_para.expand(batch_size, -1).detach(), sel_exp_para, model_3dmm.rigid_ids)
                rott_geo_lap = forward_rott(geometry_lap,  sel_euler, sel_trans)
                loss_lap = cal_lap_loss([rott_geo_lap.reshape(rott_geo_lap.shape[0], -1).permute(1, 0)],
                                        [1.0])

            loss = loss_col*0.5 + loss_lan*8 + loss_lap*100000 + loss_regexp*1.0
            if iter > 30:
                loss = loss_col*0.5 + loss_lan*1.5 + loss_lap*100000 + loss_regexp*1.0
            optimizer_cur_batch.zero_grad()
            loss.backward()
            optimizer_cur_batch.step()

        print(str(i) + ' of ' + str(int((num_frames-1)/batch_size+1)) + ' done')
        render_proj = sel_imgs.clone()
        render_proj[mask] = render_imgs[mask][..., :3].byte()
        # debug_render_dir = os.path.join(id_dir, 'debug', 'debug_render')
        # Path(debug_render_dir).mkdir(parents=True, exist_ok=True)
        # for j in range(sel_ids.shape[0]):
        #     img_arr = render_proj[j, :, :, :3].byte().detach().cpu().numpy()[
        #         :, :, ::-1]
        #     cv2.imwrite(os.path.join(debug_render_dir, str(sel_ids[j]) + '.jpg'),
        #                 img_arr)


        exp_para[sel_ids] = sel_exp_para.clone()
        euler_angle[sel_ids] = sel_euler.clone()
        trans[sel_ids] = sel_trans.clone()
        light_para[sel_ids] = sel_light.clone()
        frames_result[sel_ids] = render_proj.detach().cpu().numpy()

    return_dict = {
        'id': id_para.detach().cpu().numpy(), 
        'exp': exp_para.detach().cpu().numpy(),
        'euler': euler_angle.detach().cpu().numpy(), 
        'trans': trans.detach().cpu().numpy(),
        'focal': focal_length.detach().cpu().numpy(),
        'imgs': frames_result
    }
    return return_dict


def main(opt):
    # os.environ['CUDA_VISIBLE_DEVICES']= opt.device_id
    torch.cuda.set_device(int(opt.device_id))

    # neglect redundent processing
    is_mat_exist = os.path.exists(opt.output_file)
    is_video_exist = os.path.exists(opt.debug_video)
    # if is_mat_exist and is_video_exist:
    #     exit(0)

    # load data
    frames = read_video(opt.input_video)
    lms = read_txt(opt.keypoint_file, len(frames))

    return_dict = run(frames, lms, opt, id_dim = 100, exp_dim = 79, tex_dim = 100, point_dim = 34650)
    imgs = return_dict.pop('imgs')

    os.makedirs('/'.join(opt.output_file.split('/')[:-1]), exist_ok=True)
    os.makedirs('/'.join(opt.debug_video.split('/')[:-1]), exist_ok=True)

    imgs = [img for img in imgs]
    savemat(opt.output_file, return_dict)
    mimsave(opt.debug_video, imgs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_video', type = str, help = 'the folder to the input file')
    parser.add_argument('--output_file', type = str, help= 'the folder to the output directory')
    parser.add_argument('--keypoint_file', type=str, help='the folder to the output files')
    parser.add_argument('--debug_video', type=str, help='the folder to the output files')
    parser.add_argument('--img_h', type=int, default=512, help='image height')
    parser.add_argument('--img_w', type=int, default=512, help='image width')
    parser.add_argument('--device_id', type=str, default='0', help='image width')   
    opt = parser.parse_args()

    main(opt)