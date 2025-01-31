import torch
from utils.TOuNN_GA import *
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def pix_diff(img1, img2):
    # 确保图像大小一致
    assert img1.shape == img2.shape, "The images must have the same dimensions."
    # 计算绝对差异
    absolute_difference = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    # 计算平均绝对差异
    pix_diff = np.mean(absolute_difference)
    return pix_diff


def remove_similar_chromosomes(population, ssim_threshold, pixdiff_threshold, compliance_threshold, vol_threshold):
    """
    剔除高度相似的个体。

    参数：
    - population: list，种群，每个元素是一个 chromosome 对象
    - ssim_threshold: float，SSIM 的阈值
    - pixdiff_threshold: float，像素差异的阈值
    - compliance_threshold: float，compliance 差异的阈值
    - vol_threshold: float，vol 差异的阈值

    返回：
    - pruned_population: list，剔除相似个体后的种群
    """
    to_remove = set()
    num_chromosomes = len(population)

    for i in range(num_chromosomes):
        if i in to_remove:
            continue

        for j in range(i + 1, num_chromosomes):
            if j in to_remove:
                continue

            # 计算形态相似度指标
            image_i = population[i].image.squeeze(0).clone().detach().cpu().numpy()
            image_j = population[j].image.squeeze(0).clone().detach().cpu().numpy()
            ssim_value = ssim(image_i, image_j, data_range=image_i.max() - image_j.min())
            pix_diff_value = pix_diff(image_i, image_j)

            # 计算优化指标差异
            compliance_diff = abs(population[i].compliance - population[j].compliance)
            vol_diff = abs(population[i].vol_constraint - population[j].vol_constraint)

            # 判断是否需要剔除
            if (
                    ((ssim_value > ssim_threshold) and (pix_diff_value < pixdiff_threshold)) and
                    ((compliance_diff < compliance_threshold) and (vol_diff < vol_threshold))
            ):
                to_remove.add(j)

    # 构建剔除后的种群
    pruned_population = [population[k] for k in range(num_chromosomes) if k not in to_remove]
    return pruned_population


def permute(chromosome, values):
    chromosome.alphaIncrement, chromosome.penalIncrement, \
        chromosome.betaIncrement, chromosome.sigmaIncrement = [np.random.choice(values[-i, :]) for i in range(4,0,-1)]

def fill_contours(ref_img):
    # 二值化处理
    _, binary_image = cv2.threshold(ref_img, 127, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 创建一个与 原图大小相同的全零矩阵
    filled_image = np.zeros_like(ref_img)
    # 将轮廓内部填充为1
    cv2.drawContours(filled_image, contours, -1, color=1, thickness=cv2.FILLED)
    return filled_image, contours

def GA(batch_size, iteration, mini_iteration, resolution_fea, resolution_image, ref_img,
       output_dir, use_GA, epsilonIncrement_input, betaIncrement_input, numFolds_input=None, lmin_input=None, dvf_input=None, numTerms_input=None, fy_input=None,
       random_seed_input=None, rhoelem_perturbation_scale=0., evaluators=None, save_log=False):

    # 允许手绘任意线条，线条作为ref_img_edge，若有封闭轮廓，作为ref_img
    if ref_img != '' and os.path.exists(ref_img):
        ref_img_raw = cv2.imread(ref_img, cv2.IMREAD_UNCHANGED)[:, :]
        # 创建ref_img_edge
        kernel = np.ones((2, 2), np.uint8)  # 设定膨胀的内核大小
        t = cv2.dilate(ref_img_raw.astype(float), kernel, iterations=1)  # 进行膨胀操作
        t = cv2.resize(t, dsize=(resolution_image, resolution_image)).astype(np.float) / 255.
        ref_img_edge_tensor = torch.tensor(t, requires_grad=True).to('cuda').float().unsqueeze(0).detach()
        # 将参考图像内的封闭轮廓线内区域填充为白色，其他区域为0（黑色），若没有则全为黑
        t, contours = fill_contours(ref_img_raw)
        print(f'封闭轮廓数：{len(contours)}')
        t = cv2.resize(t, dsize=(resolution_image, resolution_image)).astype(np.float)
        t = torch.tensor(t, requires_grad=True).to('cuda').float()
        ref_img_tensor = t.unsqueeze(0).detach()
    else:
        ref_img_tensor = None
        ref_img_edge_tensor = None


    # 初始化种群
    values = np.array([
    np.linspace(0, 1.5, num=200), # fy
    np.linspace(500, 1100, num=200), # numTerms
    np.linspace(0.01, 0.01, num=200), # learning_rate
    np.linspace(80, 80, num=200), # lmax
    np.linspace(3, 9, num=200), # sharpness
    np.linspace(1, 1, num=200), # num_layers
    np.linspace(100, 100, num=200), # num_neurons_per_layer
    np.linspace(0.2, 0.6, num=200), # alphaIncrement
    np.linspace(0.05, 0.1, num=200), # penalIncrement
    np.linspace(2e-5, 1e-4, num=200), # betaIncrement
    np.linspace(5e-4, 5e-3, num=200),  # sigmaIncrement
    np.linspace(5e-6, 5e-5, num=200),  # epsilonIncrement
    np.linspace(0.25, 0.55, num=200),  # dvf
    np.linspace(5, 12, num=200),       # lmin
    np.linspace(3, 10, num=200)        # numFolds
    ])

    chromosomes = []
    for i in range(batch_size):

        random_seed = random_seed_input if random_seed_input is not None else i
        np.random.seed(random_seed)

        # 每个个体随机取值
        fy, numTerms, learning_rate, lmax, sharpness, num_layers, num_neurons_per_layer, \
        alphaIncrement, penalIncrement, betaIncrement, sigmaIncrement, epsilonIncrement, dvf, lmin, numFolds = \
            [np.random.choice(values[i, :]) for i in range(values.shape[0])]

        sharpness=5; alphaIncrement=0.3
        penalIncrement=0.04
        # fy = 1.
        # numTerms = 1000
        # betaIncrement=5e-5
        sigmaIncrement=0.
        epsilonIncrement = epsilonIncrement_input
        betaIncrement = betaIncrement_input

        if numTerms_input is not None:
            numTerms = int(numTerms_input)
        else:
            numTerms = int(numTerms)

        if fy_input is not None:
            fy = fy_input

        if dvf_input is not None:
            dvf = dvf_input

        if lmin_input is not None:
            lmin = int(lmin_input)
        else:
            lmin = int(lmin)

        if numFolds_input is not None:
            numFolds = int(numFolds_input)
        else:
            numFolds = int(numFolds)

        # 初始化个体
        print(f"resolution_fea: {resolution_fea}")
        print(f"resolution_image: {resolution_image}")
        print(f"dvf: {dvf}")
        print(f"numFolds: {numFolds}")
        print(f"ref_img: {ref_img}")
        print(f"lmin: {lmin}")
        print(f"fy: {fy}")
        print(f"numTerms: {numTerms}")
        print(f"learning_rate: {learning_rate}")
        print(f"lmax: {lmax}")
        print(f"sharpness: {sharpness}")
        print(f"num_layers: {num_layers}")
        print(f"num_neurons_per_layer: {num_neurons_per_layer}")
        print(f"alphaIncrement: {alphaIncrement}")
        print(f"penalIncrement: {penalIncrement}")
        print(f"betaIncrement: {betaIncrement}")
        print(f"sigmaIncrement: {sigmaIncrement}")
        print(f"epsilonIncrement: {epsilonIncrement}")

        chromosome = TopologyOptimizer(
            resolution_fea=resolution_fea,  # 各个拓扑优化问题的一致的参数
            resolution_image=resolution_image,
            dvf=dvf,
            numFolds=numFolds,
            ref_img_tensor=ref_img_tensor,
            ref_img_edge_tensor=ref_img_edge_tensor,
            lmin=lmin,
            fy=fy,                        # 各个拓扑优化问题的差异化的参数
            numTerms=numTerms,
            learning_rate=learning_rate,
            lmax=lmax,
            sharpness=sharpness,
            num_layers=num_layers,
            num_neurons_per_layer=num_neurons_per_layer,
            alphaIncrement=alphaIncrement,
            penalIncrement=penalIncrement,
            betaIncrement=betaIncrement,
            sigmaIncrement=sigmaIncrement,
            epsilonIncrement=epsilonIncrement,
            random_seed=random_seed,
            name='',
            output_dir=output_dir,
            save_log=save_log,
            rhoelem_perturbation_scale=rhoelem_perturbation_scale
        )
        chromosomes.append(chromosome)
    chromosomes = np.array(chromosomes)



    start = time.time()
    for iter in range(iteration):
        print("Iteration: ", iter)
        # 每运行TO算法的迭代5次，种群繁衍一代
        for chromosome in chromosomes:
            chromosome.opimize(mini_iteration)

        # 保存当代结果
        output_dir_this_iter = os.path.join(output_dir, f'iteration_{iter}')
        os.mkdir(output_dir_this_iter)
        for i, chromosome in enumerate(chromosomes):
            compliance = chromosome.compliance
            vol = chromosome.vol_constraint
            von_mises_stress = chromosome.compute_von_mises_stress(chromosome.FE.mesh.material['E'],
                                                                  chromosome.FE.mesh.material['nu'],
                                                                  chromosome.FE.mesh.material['penal'],
                                                                  chromosome.density,
                                                                  chromosome.uElem)
            chromosome.save_image(chromosome.image * 255.,
                                  filepath=os.path.join(output_dir_this_iter, f'dvf_{chromosome.dvf}_{i}_comp_{compliance:.7f}_vol_{vol:.7f}.png'))
            chromosome.plot_vonMises_stress(von_mises_stress, field_density=None,
                                      save_filename=os.path.join(output_dir_this_iter, f'{i}_von_mises_stress.png'))

        if iter > 10 and use_GA:
            # 根据形态相似度和优化目标相似度，筛除高度相似的个体
            chromosomes = remove_similar_chromosomes(
                population=chromosomes,
                ssim_threshold=0.9,
                pixdiff_threshold=0.02,
                compliance_threshold=0.1,
                vol_threshold=0.01
            )

    print('time ellapsed: ', time.time()-start)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="GA Optimization Parameters")

    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for GA')
    parser.add_argument('--iteration', type=int, required=True, help='Number of GA iterations')
    parser.add_argument('--mini_iteration', type=int, required=True, help='Number of mini iterations per GA iteration')
    parser.add_argument('--resolution_fea', type=int, required=True, help='Resolution for feature analysis')
    parser.add_argument('--resolution_image', type=int, required=True, help='Resolution for image analysis')
    parser.add_argument('--dvf_input', type=float, help='DVF parameter for GA')
    parser.add_argument('--numFolds_input', type=int, help='Number of folds for cross-validation')
    parser.add_argument('--use_ref_img', type=str, required=True, help='reference image')
    parser.add_argument('--lmin_input', type=float, help='Minimum length scale')
    parser.add_argument('--work_dir', type=str, required=True, help='Directory for output')
    parser.add_argument('--use_GA', type=str, required=True, help='use_GA')
    parser.add_argument('--epsilonIncrement_input', type=float, required=True, help='epsilonIncrement_input')
    parser.add_argument('--betaIncrement_input', type=float, required=True, help='betaIncrement_input')
    parser.add_argument('--numTerms_input', type=int, help=' ')
    parser.add_argument('--random_seed_input', type=int, help=' ')
    parser.add_argument('--fy_input', type=float, help=' ')
    parser.add_argument('--rhoelem_perturbation_scale', type=float, default=0., help=' ')
    parser.add_argument('--save_log')

    args = parser.parse_args()

    assert os.path.isdir(args.work_dir)
    output_dir = os.path.join(args.work_dir, 'samples','2d')

    if args.use_ref_img == 'True':
        ref_img = os.path.join(args.work_dir, 'sketch.png')
        assert os.path.exists(ref_img)
    else:
        ref_img = ''

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    GA(
        batch_size=args.batch_size,
        iteration=args.iteration,
        mini_iteration=args.mini_iteration,
        resolution_fea=args.resolution_fea,
        resolution_image=args.resolution_image,
        dvf_input=args.dvf_input,
        numFolds_input=args.numFolds_input,
        ref_img=ref_img,
        lmin_input=args.lmin_input,
        output_dir=output_dir,
        use_GA=args.use_GA=='True',
        epsilonIncrement_input=args.epsilonIncrement_input,
        betaIncrement_input=args.betaIncrement_input,
        numTerms_input=args.numTerms_input,
        fy_input=args.fy_input,
        random_seed_input=args.random_seed_input,
        rhoelem_perturbation_scale=args.rhoelem_perturbation_scale,
        save_log=args.save_log=='True'
    )
