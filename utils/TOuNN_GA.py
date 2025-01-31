import os
import numpy as np
import torch
from utils.FE import *
from utils.network import *
import torch.optim as optim
import cv2
import torch.nn.functional as F
import shutil
import time
import matplotlib.pyplot as plt



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# np.random.seed(0)

def inside_circle(x, y, center, r):
    return (np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) < r)

def to_np(x):
    return x.clone().detach().cpu().numpy()


class AddRandomNoiseToGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 保存输入数据，以便在反向传播中使用
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # 获取梯度的最大绝对值作为参考范围
        grad_abs_max = grad_output.abs().max()

        # 设定噪声幅度：噪声是梯度最大绝对值的 10%
        noise = torch.rand_like(grad_output) * 1.  # 随机噪声范围是 [0, grad_abs_max * 0.1]

        # 在梯度中添加随机噪声
        grad_input = grad_output + noise

        return grad_input


# 单个拓扑优化问题
class TopologyOptimizer:

    def __init__(self,
                 resolution_fea, resolution_image, dvf, numFolds, lmin,  # 各个拓扑优化问题的一致的参数
                 fy, numTerms, learning_rate, lmax, sharpness,  # 各个拓扑优化问题的差异化的参数
                 num_layers, num_neurons_per_layer, alphaIncrement, penalIncrement,
                 betaIncrement, sigmaIncrement, epsilonIncrement, random_seed=None,
                 name='', output_dir=None, save_log=False, ref_img_tensor=None, ref_img_edge_tensor=None,
                 rhoelem_perturbation_scale=0.):

        self.rhoElem_perturbation_scale = rhoelem_perturbation_scale
        assert isinstance(random_seed, int)
        np.random.seed(random_seed)

        assert output_dir is not None
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        self.output_dir = output_dir
        self.intermediate_dir = os.path.join(output_dir, f'{name}_intermediate')
        if save_log:
            os.mkdir(self.intermediate_dir)
            self.optimize_log = {}
            self.optimize_log['similarity_edge'] = []
            self.optimize_log['similarity_void'] = []
            self.optimize_log['vol_constraint'] = []
            self.optimize_log['compliance'] = []

        self.save_log = save_log
        self.save_filename = os.path.join(output_dir, name+'.png')

        self.alphaIncrement = alphaIncrement
        self.penalIncrement = penalIncrement
        self.betaIncrement = betaIncrement
        self.sigmaIncrement = sigmaIncrement
        self.epsilonIncrement = epsilonIncrement

        self.alpha = alphaIncrement
        self.penal = 1.
        self.beta = betaIncrement
        self.sigma = sigmaIncrement
        self.epsilon = epsilonIncrement

        self.resolution_image = resolution_image
        self.resolution_fea = resolution_fea
        self.keepElems_fea = self.compute_keepElems(nelx=resolution_fea, nely=resolution_fea)
        self.keepElems_image = self.compute_keepElems(nelx=resolution_image, nely=resolution_image)

        mesh_fea = {'type': 'grid', 'nelx': resolution_fea, 'nely': resolution_fea, 'elemSize': np.array([1.0, 1.0])}
        mesh_image = {'type': 'grid', 'nelx': resolution_image, 'nely': resolution_image, 'elemSize': np.array([1.0, 1.0])}

        bc_fea = {'exampleName': 'TO', 'physics': 'Structural', 'numDOFPerNode': 2,
                  'force': self.compute_force(resolution_fea, resolution_fea, fy),
                  'fixed': self.compute_fixed(resolution_fea, resolution_fea)}
        matProp = {'E': 1.0, 'nu': 0.3, 'penal': self.penal}

        self.FE = FE(mesh_fea, matProp, bc_fea)
        self.symCentric_fea = {'isOn': True, 'numFolds': numFolds,
                          'center': [resolution_fea / 2, resolution_fea / 2]}
        self.symCentric_image = {'isOn': True, 'numFolds': numFolds,
                            'center': [resolution_image / 2, resolution_image / 2]}

        xy_fea = self.FE.mesh.generatePoints(resolution=1)
        xy_image = GridMesh(mesh_image, matProp, bc=None).generatePoints(resolution=1)
        self.xy_fea = torch.tensor(xy_fea, requires_grad=True).float().view(-1, 2).to('cuda')
        self.xy_image = torch.tensor(xy_image, requires_grad=True).float().view(-1, 2).to('cuda')
        self.learning_rate = learning_rate
        self.dvf = dvf

        self.densityProjection = {'isOn': True, 'sharpness': sharpness}
        self.fourierMap = {'isOn': True, 'minRadius': lmin,
                      'maxRadius': lmax, 'numTerms': numTerms}
        if (self.fourierMap['isOn']):
            coordnMap = np.zeros((2, int(self.fourierMap['numTerms'])))
            for i in range(coordnMap.shape[0]):
                for j in range(coordnMap.shape[1]):
                    coordnMap[i, j] = np.random.choice([-1., 1.]) * np.random.uniform(
                        1. / (2 * lmax), 1. / (2 * lmin))
            self.coordnMap = torch.tensor(coordnMap).float().to('cuda')
            inputDim = 2 * self.coordnMap.shape[1]
        else:
            self.coordnMap = torch.eye(2)
            inputDim = 2

        nnSettings = {'numLayers': int(num_layers), 'numNeuronsPerLyr': int(num_neurons_per_layer)}
        self.topNet = TopNet(nnSettings, inputDim).to('cuda')
        self.density = self.dvf * np.ones((self.FE.mesh.numElems))
        self.optimizer = optim.Adam([
                {
                    'params': self.topNet.parameters(),
                    'lr': learning_rate
                }])

        self.ref_img_tensor = ref_img_tensor
        self.ref_img_edge_tensor = ref_img_edge_tensor

        self.penalMax = 6.
        self.alphaMax = 100
        self.betaMax = 1
        self.sigmaMax = 2.
        self.epsilonMax = 0.1

        self.compliance = 0.
        self.vol_constraint = 0.
        self.similarity_loss = 0.
        self.saliency_loss = 0.

        self.image = None

        self.obj0 = None


    def compute_B_matrix(self, x, y, xi, eta):
        # 形函数的局部坐标导数
        dN_dxi = np.array([
            [-0.25 * (1 - eta), -0.25 * (1 - xi)],
            [0.25 * (1 - eta), -0.25 * (1 + xi)],
            [0.25 * (1 + eta), 0.25 * (1 + xi)],
            [-0.25 * (1 + eta), 0.25 * (1 - xi)]
        ])

        # 计算雅可比矩阵
        J = np.dot(dN_dxi.T, np.vstack([x, y]).T)

        # 雅可比矩阵的逆
        J_inv = np.linalg.inv(J)

        # 形函数相对于全局坐标的导数
        dN_dxy = np.dot(dN_dxi, J_inv)

        # 构造B矩阵
        B = np.zeros((3, 8))
        B[0, 0::2] = dN_dxy[:, 0]  # 对应dN/dx
        B[1, 1::2] = dN_dxy[:, 1]  # 对应dN/dy
        B[2, 0::2] = dN_dxy[:, 1]  # 对应dN/dy
        B[2, 1::2] = dN_dxy[:, 0]  # 对应dN/dx

        return B


    def compute_D_matrix(self, E, nu):
        # 计算材料弹性矩阵
        D = (E / (1 - nu ** 2)) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])
        return D

    def compute_Ke(self, E, nu, x, y):
        # 材料弹性矩阵 D
        D = self.compute_D_matrix(E, nu)

        # 高斯积分点（2x2 Gauss points）
        gauss_points = [(-0.577, -0.577), (0.577, -0.577), (-0.577, 0.577), (0.577, 0.577)]
        gauss_weights = [1, 1, 1, 1]

        # 初始化刚度矩阵
        Ke = np.zeros((8, 8))

        for (xi, eta), w in zip(gauss_points, gauss_weights):
            # 计算应变-位移矩阵 B
            B = self.compute_B_matrix(x, y, xi, eta)

            # 局部刚度矩阵 dK = B.T @ D @ B * |J| * w
            # 其中 |J| 是雅可比矩阵的行列式，w 是高斯积分权重
            J = np.dot(np.array([
                [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)],
                [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)]
            ]), np.vstack([x, y]).T)
            detJ = np.linalg.det(J)

            # 局部刚度矩阵
            Ke += B.T @ D @ B * detJ * w

        return Ke


    def compute_B_matrix2(self, xi, eta):
        # 形函数的局部坐标导数
        dN_dxi = np.array([
            [-0.25 * (1 - eta), -0.25 * (1 - xi)],
            [0.25 * (1 - eta), -0.25 * (1 + xi)],
            [0.25 * (1 + eta), 0.25 * (1 + xi)],
            [-0.25 * (1 + eta), 0.25 * (1 - xi)]
        ])

        # 构造B矩阵
        B = np.zeros((3, 8))
        B[0, 0::2] = dN_dxi[:, 0]  # 对应dN/dx
        B[1, 1::2] = dN_dxi[:, 1]  # 对应dN/dy
        B[2, 0::2] = dN_dxi[:, 1]  # 对应dN/dy
        B[2, 1::2] = dN_dxi[:, 0]  # 对应dN/dx

        return B * 2

    def compute_von_mises_stress(self, E, nu, penal, density, uElem):
        # 高斯积分点（2x2 Gauss points）, 通常四节点单元有4个高斯积分点
        gauss_points = [(-0.577, -0.577), (0.577, -0.577), (-0.577, 0.577), (0.577, 0.577)]
        gauss_weights = [1, 1, 1, 1]

        # 初始化应力数组 (假设平面应力问题)
        num_elements = len(uElem)
        von_mises_stress = np.zeros(num_elements)  # 每个单元一个冯·米塞斯应力值

        # 对每个单元计算应力
        for i in range(num_elements):
            # 获取该单元的位移向量
            u = uElem[i]

            # 构建材料的弹性矩阵D (平面应力情况)，惩罚
            E_elem = E * (0.01 + density[i]) ** penal
            D = self.compute_D_matrix(E_elem, nu)

            # 初始化单元内的冯·米塞斯应力累加
            von_mises_sum = 0

            # 对该单元的每个高斯积分点计算应力
            for (xi, eta), w in zip(gauss_points, gauss_weights):
                # 获取B矩阵（在每个积分点处）
                B = self.compute_B_matrix2(xi, eta)
                # 计算应变
                strain = np.dot(B, u)
                # 计算应力
                stress = np.dot(D, strain)
                # 计算该积分点处的冯·米塞斯应力
                sigma_xx = stress[0]
                sigma_yy = stress[1]
                sigma_xy = stress[2]

                von_mises = np.sqrt(sigma_xx ** 2 - sigma_xx * sigma_yy + sigma_yy ** 2 + 3 * sigma_xy ** 2)

                # 累加每个积分点的冯·米塞斯应力
                von_mises_sum += von_mises

            # 将该单元的冯·米塞斯应力取积分点的平均值
            von_mises_stress[i] = von_mises_sum / len(gauss_weights)

        return von_mises_stress


    def sobel_edge_detection(self, image):
        # 定义 Sobel 卷积核
        sobel_kernel_x = torch.tensor([[-1., 0., 1.],
                                       [-2., 0., 2.],
                                       [-1., 0., 1.]]).view(1, 1, 3, 3).to(dtype=torch.float32).to('cuda')

        sobel_kernel_y = torch.tensor([[-1., -2., -1.],
                                       [0., 0., 0.],
                                       [1., 2., 1.]]).view(1, 1, 3, 3).to(dtype=torch.float32).to('cuda')

        # 对图像应用 Sobel 卷积核
        grad_x = F.conv2d(image, sobel_kernel_x, padding=1)
        grad_y = F.conv2d(image, sobel_kernel_y, padding=1)

        # 计算梯度的幅值
        l2 = grad_x ** 2 + grad_y ** 2 + 1e-6
        grad_magnitude = torch.sqrt(l2)

        grad_magnitude.retain_grad()
        grad_x.retain_grad()
        grad_y.retain_grad()

        min_val = grad_magnitude.min()
        max_val = grad_magnitude.max()
        normalized = (grad_magnitude - min_val) / (max_val - min_val)

        return normalized, grad_magnitude, grad_x, grad_y, l2


    def compute_force(self, nelx, nely, fy, F=1):
        ndof = 2 * (nelx + 1) * (nely + 1)
        force = np.zeros((ndof, 1))
        for theta in range(-14, 16, 2):
            x = nelx // 2 - int((nelx // 2) * np.cos(np.deg2rad(theta))) + 2
            y = nely // 2 - int((nely // 2) * np.sin(np.deg2rad(theta)))
            F_x = F * np.cos(np.deg2rad(theta))
            F_y = F * np.sin(np.deg2rad(theta)) + fy
            force[2 * x * (nely + 1) + 2 * y, 0] = F_x
            force[2 * x * (nely + 1) + 2 * y + 1, 0] = F_y
        return force


    def compute_fixed(self, nelx, nely):
        fixed = []
        for theta in np.linspace(0, 2 * np.pi, 5000):
            x = int(0.13 * nelx * np.cos(theta)) + nelx // 2  # 30/230*100=13
            y = int(0.13 * nely * np.sin(theta)) + nely // 2
            fixed.extend([2 * (nely + 1) * x + 2 * y, 2 * (nely + 1) * x + 2 * y + 1])
        fixed = np.unique(np.array(fixed))
        return fixed


    def compute_keepElems(self, nelx, nely):
        elems = np.array([], dtype=np.int32)
        for x in range(nelx):
            for y in range(nely):
                if (not inside_circle(x + 0.5, y + 0.5, center=[nelx / 2, nely / 2], r=nelx / 2 - 0.5)) or \
                        inside_circle(x + 0.5, y + 0.5, center=[nelx / 2, nely / 2], r=0.13 * nelx - 0.5):
                    elems = np.append(elems, nely * x + y)
        keepElems = [{'idx': elems, 'density': 0.01}]

        elems = np.array([], dtype=np.int32)
        for x in range(nelx):
            for y in range(nely):
                if inside_circle(x + 0.5, y + 0.5, center=[nelx / 2, nely / 2], r=nelx / 2 - 0.5) and \
                        (not inside_circle(x + 0.5, y + 0.5, center=[nelx / 2, nely / 2], r=0.43 * nelx)):
                    elems = np.append(elems, nely * x + y)
        keepElems += [{'idx': elems, 'density': 0.99}]

        elems = np.array([], dtype=np.int32)
        for x in range(nelx):
            for y in range(nely):
                if inside_circle(x + 0.5, y + 0.5, center=[nelx / 2, nely / 2], r=0.17 * nelx + 0.5) and \
                        (not inside_circle(x + 0.5, y + 0.5, center=[nelx / 2, nely / 2], r=0.13 * nelx - 0.5)):
                    elems = np.append(elems, nely * x + y)
        keepElems += [{'idx': elems, 'density': 0.99}]
        return keepElems


    def applyCentricSymmetry(self, x, symCentric):
        if (symCentric['isOn']):
            theta = 180. / np.pi * torch.atan2(x[:, 1] - symCentric['center'][1],
                                               x[:, 0] - symCentric['center'][0])
            r = torch.sqrt(
                (x[:, 1] - symCentric['center'][1]) ** 2 + (x[:, 0] - symCentric['center'][0]) ** 2)
            theta_span = 360. / symCentric['numFolds']

            theta -= 180 - theta_span / 2
            theta[theta < 0] += 360
            theta = theta % theta_span + 180 - theta_span / 2

            xv = r * torch.cos((theta) / 180 * np.pi)  # 先前为方便计算，将角度范围[-pi,pi]转为[0,2*pi]
            yv = r * torch.sin((theta) / 180 * np.pi)
            xv += symCentric['center'][0]
            yv += symCentric['center'][1]
            x = torch.transpose(torch.stack((xv, yv)), 0, 1)
            x[x < 0] = 0.

        return x


    def applyFourierMapping(self, x):
        if (self.fourierMap['isOn']):
            c = torch.cos(2 * np.pi * torch.matmul(x, self.coordnMap))
            s = torch.sin(2 * np.pi * torch.matmul(x, self.coordnMap))
            xv = torch.cat([c, s], dim=1)
            return xv
        return x

    def projectDensity(self, x):
        if (self.densityProjection['isOn']):
            b = self.densityProjection['sharpness']
            nmr = np.tanh(0.5 * b) + torch.tanh(b * (x - 0.5))
            x = 0.5 * nmr / np.tanh(0.5 * b)
        return x

    def to_image_no_interpolate(self, rhoElem):
        width = int(np.sqrt(rhoElem.size(0)))
        image = rhoElem.reshape((width, width)).float()

        return 1. - image

    def save_image(self, tensor_image, filepath):
        # 将 PyTorch tensor 转为 numpy 数组
        numpy_image = tensor_image.clone().detach().cpu().numpy().astype(np.uint8)
        numpy_image = np.transpose(numpy_image, [1, 2, 0])
        # 保存为灰度图像
        cv2.imwrite(filepath, numpy_image)


    def plot_vonMises_stress(self, von_mises_stress, field_density, save_filename):
        von_mises_stress = von_mises_stress.reshape(self.FE.mesh.nelx, self.FE.mesh.nely)

        tensor_clipped = von_mises_stress
        # 将张量归一化到 0-255 的范围
        normalized_tensor = cv2.normalize(tensor_clipped, None, 0, 255, cv2.NORM_MINMAX)
        # 将归一化后的张量转换为 8 位无符号整数
        normalized_tensor = normalized_tensor.astype(np.uint8)
        # 将灰度图转换为彩色图，使用 'JET' 颜色映射
        color_map = cv2.applyColorMap(normalized_tensor, cv2.COLORMAP_JET)

        # 只保留有材料的区域的色彩，空白区域为白色
        if field_density is not None:
            field_density = field_density.reshape(self.FE.mesh.nelx, self.FE.mesh.nely)
            void_region = field_density < 0.15
            color_map[void_region] = (255.,255.,255.)

        # 保存彩色图片
        cv2.imwrite(save_filename, color_map)


    def remove_rim_and_hub_circles(self, img, min_radius=38., max_radius=100.):
        c, h, w = img.shape
        center_y, center_x = h // 2, w // 2
        # 生成一个与图像大小相同的网格
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        # 将网格坐标转换为浮点类型
        y = y.float()
        x = x.float()
        # 计算网格上每个点到图像中心的距离
        distance = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        # 找到满足条件的区域
        mask = (distance < min_radius) | (distance > max_radius)
        # 将这些区域填充为 0
        mask = mask.unsqueeze(0)
        img[mask] = 0.0
        return img

    # 迭代优化指定的epoch代
    def opimize(self, epoch):
        for _ in range(epoch):

            self.optimizer.zero_grad()

            x_after_ft_mapping_fea = self.applyFourierMapping(
                self.applyCentricSymmetry(self.xy_fea, self.symCentric_fea))
            x_after_ft_mapping_image = self.applyFourierMapping(
                self.applyCentricSymmetry(self.xy_image, self.symCentric_image))
            input = torch.cat([x_after_ft_mapping_fea, x_after_ft_mapping_image], dim=0)
            split = x_after_ft_mapping_fea.size(0)

            nn_rho = torch.flatten(self.topNet(input)).to('cuda')
            rhoElem = self.projectDensity(nn_rho)
            noise = torch.rand_like(rhoElem, requires_grad=False) * self.rhoElem_perturbation_scale
            rhoElem = torch.clamp(rhoElem + noise, min=0.0, max=1.0)


            rhoElem_fea = rhoElem[:split]
            rhoElem_image = rhoElem[split:]

            res = int(np.sqrt(rhoElem_fea.size(0)))
            rhoElem_fea = rhoElem_fea.reshape((res, res)).flatten()
            for i in range(len(self.keepElems_fea)):
                rhoElem_fea[self.keepElems_fea[i]['idx']] = self.keepElems_fea[i]['density']

            res = int(np.sqrt(rhoElem_image.size(0)))
            rhoElem_image = rhoElem_image.reshape((res, res)).flatten()
            for i in range(len(self.keepElems_image)):
                rhoElem_image[self.keepElems_image[i]['idx']] = self.keepElems_image[i]['density']

            density = to_np(rhoElem_fea)

            u, Jelem, uElem = self.FE.solve(density)  # Call FE 88 line code [Niels Aage 2013]

            self.density = density
            self.uElem = uElem

            if self.obj0 is None:
                self.obj0 = ((0.01 + density) ** (2 * self.FE.mesh.material['penal']) * Jelem).sum()

            # For sensitivity analysis, exponentiate by 2p here and divide by p in the loss func hence getting -ve sign
            Jelem = np.array((density ** (2 * self.FE.mesh.material['penal'])) * Jelem).reshape(-1)
            Jelem = torch.tensor(Jelem).view(-1).float().to(torch.device('cuda:0'))
            compliance = torch.sum(
                torch.div(Jelem, rhoElem_fea ** self.FE.mesh.material['penal'])) / self.obj0  # compliance
            if self.save_log:
                self.optimize_log['compliance'].append(compliance.clone().item())

            volConstraint = ((torch.sum(self.FE.mesh.elemArea.to('cuda') * rhoElem_fea) /
                              (self.FE.mesh.netArea.to(
                                  'cuda') * self.dvf)) - 1.0)  # global vol constraint
            vol_constraint = self.alpha * torch.pow(volConstraint, 2)
            if self.save_log:
                self.optimize_log['vol_constraint'].append(vol_constraint.clone().item())

            image = self.to_image_no_interpolate(rhoElem_image).unsqueeze(0).to('cuda')
            self.image = image

            inner_r = 39.2 / 256. * (self.resolution_image)
            inner_rim_r = 125. / 256. * (self.resolution_image)
            similarity_loss = torch.tensor(0.).to('cuda')
            similarity_void = torch.tensor(0.).to('cuda')
            similarity_edge = torch.tensor(0.).to('cuda')
            saliency_loss = torch.tensor(0.).to('cuda')
            if (self.ref_img_tensor is not None) and (self.ref_img_edge_tensor is not None):
                o = self.sobel_edge_detection(image.unsqueeze(0))

                edge = o[0].squeeze(0)

                edge = self.remove_rim_and_hub_circles(edge, min_radius=inner_r + 2., max_radius=inner_rim_r - 2.)


                diff_img = torch.where(self.ref_img_tensor > 0.5, (self.ref_img_tensor - image) ** 2, torch.zeros_like(image))
                diff_edge = torch.where(self.ref_img_edge_tensor > 0.5, (self.ref_img_edge_tensor - edge) ** 2,
                                        torch.zeros_like(edge))
                similarity_void += self.epsilon * torch.sum(diff_img)
                similarity_edge += self.beta * torch.sum(diff_edge)
                similarity_loss += self.epsilon * torch.sum(diff_img) + self.beta * torch.sum(diff_edge)

                similarity_edge_item = similarity_edge.clone().detach().cpu().item()
                similarity_void_item = similarity_void.clone().detach().cpu().item()
                compliance_item = compliance.clone().detach().cpu().item()
                vol_constraint_item = vol_constraint.clone().detach().cpu().item()

                if self.save_log:
                    self.optimize_log['similarity_edge'].append(similarity_edge_item)
                    self.optimize_log['similarity_void'].append(similarity_void_item)

                print('similarity_edge: ', similarity_edge_item)
                print('similarity_void: ', similarity_void_item)
                print('compliance: ', compliance_item)
                print('vol_constraint: ', vol_constraint_item)

            self.compliance = compliance.clone().item()
            self.vol_constraint = vol_constraint.clone().item() / self.alpha
            self.vol_constraint = (np.sqrt(self.vol_constraint) + 1) * self.dvf
            self.similarity_loss = similarity_void.clone().item() / self.epsilon + similarity_edge.clone().item() / self.beta
            self.saliency_loss = saliency_loss.clone().item()

            loss = compliance + vol_constraint + similarity_loss # + saliency_loss

            loss.backward()

            self.optimizer.step()

            self.FE.mesh.material['penal'] = min(self.penalMax, self.FE.mesh.material[
                'penal'] + self.penalIncrement)  # continuation scheme

            self.alpha = min(self.alphaMax, self.alpha + self.alphaIncrement)
            self.beta = min(self.betaMax, self.beta + self.betaIncrement)
            self.sigma = min(self.sigmaMax, self.sigma + self.sigmaIncrement)
            self.epsilon = min(self.epsilonMax, self.epsilon + self.epsilonIncrement)

            # self.von_mises_stress = self.compute_von_mises_stress(self.FE.mesh.material['E'],
            #                                             self.FE.mesh.material['nu'],
            #                                             self.FE.mesh.material['penal'],
            #                                             density,
            #                                             uElem)


