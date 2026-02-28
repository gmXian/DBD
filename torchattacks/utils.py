import inspect
from . import *

ATTACKS_FULL = [
    # "VANILA",
    # "GN",
    "FGSM",
    "BIM",
    "RFGSM",
    "PGD",
    "ESPGD",
    "EOTPGD",
    "FFGSM",
    "TPGD",
    "MIFGSM",
    "UPGD",
    "APGD",
    "APGDT",
    "DIFGSM",
    "TIFGSM",
    "Jitter",
    "NIFGSM",
    "PGDRS",
    "SINIFGSM",
    "VMIFGSM",
    "VNIFGSM",
    "SPSA",
    "JSMA",
    "EADL1",
    "EADEN",
    "PIFGSM",
    "PIFGSMPP",
    "CW",
    "CWL0",
    "CWLinf",
    "CWBS",
    "CWBSL0",
    "CWBSLinf",
    "PGDL2",
    "DeepFool",
    "PGDRSL2",
    "SparseFool",
    "OnePixel",
    "Pixle",
    "FAB",
    "FABL1",
    "FABL2",
    "AFAB",
    "AutoAttack",
    "Square",
    "FMN",
    "ZeroGrad",
    # "MultiAttack",
    # "LGV",
]


ATTACKS_VALID = [
    # ============================================== Linf
    "FGSM",
    "BIM",
    "RFGSM",
    "PGD",
    # "ESPGD",  # not work
    # "EOTPGD", # =PGD
    # "FFGSM",    # <FGSM
    # "TPGD",     # <FGSM
    "MIFGSM",   
    "UPGD",
    # "DIFGSM",  weak
    # "TIFGSM", # bug
    "Jitter",
    "NIFGSM",
    # "PGDRS",  # not work
    # "SINIFGSM", # <FGSM
    "VMIFGSM",     # slow 5, =PGD
    "VNIFGSM",     # slow 5, >FGSM, <PGD
    # "SPSA",   # not work
    "PIFGSM",   # >FGSM, <PGD
    "PIFGSMPP", # >FGSM, <PGD
    "CWLinf", 
    "CWBSLinf", # slow 8
    # "FAB",    # out of memory
    # "ZeroGrad",   # bug
    # ============================================== L0
    # "JSMA",   # bug
    "CWL0",  
    "CWBSL0",   # slow 8
    # "SparseFool", # too slow
    # "OnePixel", # not work
    # "Pixle",  # too weak
    # ============================================== L1
    "EADL1",  # slow 12
    "EADEN",  # slow 12
    # "FABL1",  # OOM
    # ============================================== L2
    "CW",  # 
    "CWBS", # slow 8
    # "PGDL2", # weak
    # "DeepFool", # too slow
    # "PGDRSL2", # not work
    # "FABL2", 
    # ============================================== AutoAttack series: Linf, L2
    # "AFAB", # too slow and weak
    # "APGD", # slow and > FGSM, < PGD
    # "APGDT", # slow 9, =PGD
    # "AutoAttack", # slow 5
    # "Square", # not work
    # L0, L1, L2, Linf
    # "FMN", not work
]


ATTACKS_TRAIN = [
    # ============================================== Linf
    "FGSM",
    "BIM",
    "RFGSM",
    "PGD",
    # "ESPGD",  # not work
    # "EOTPGD", # =PGD
    # "FFGSM",    # <FGSM
    # "TPGD",     # <FGSM
    "MIFGSM",   
    "UPGD",
    # "DIFGSM",  weak
    # "TIFGSM", # bug
    "Jitter",
    # "NIFGSM",
    # "PGDRS",  # not work
    # "SINIFGSM", # <FGSM
    "VMIFGSM",     # slow 5, =PGD
    # "VNIFGSM",     # slow 5, >FGSM, <PGD
    # "SPSA",   # not work
    "PIFGSM",   # >FGSM, <PGD
    "PIFGSMPP", # >FGSM, <PGD
    "CWLinf", 
    "CWBSLinf", # slow 8
    # "FAB",    # out of memory
    # "ZeroGrad",   # bug
    # ============================================== L0
    # "JSMA",   # bug
    "CWL0",  
    "CWBSL0",   # slow 8
    # "SparseFool", # too slow
    # "OnePixel", # not work
    # "Pixle",  # too weak
    # ============================================== L1
    # "EADL1",  # slow 12
    "EADEN",  # slow 12
    # "FABL1",  # OOM
    # ============================================== L2
    "CW",  # 
    "CWBS", # slow 8
    # "PGDL2", # weak
    # "DeepFool", # too slow
    # "PGDRSL2", # not work
    # "FABL2", 
    # ============================================== AutoAttack series: Linf, L2
    # "AFAB", # too slow and weak
    # "APGD", # slow and > FGSM, < PGD
    # "APGDT", # slow 9, =PGD
    # "AutoAttack", # slow 5
    # "Square", # not work
    # L0, L1, L2, Linf
    # "FMN", not work
]



# generate_yaml_from_class()

def generate_class_init_call_code(cls, instance_name="instance", model_var_name="model"):
    """
    自动生成类的初始化调用代码字符串。
    
    参数:
        cls: 类对象
        instance_name: 实例变量名，如 "pgd"
        model_var_name: 模型变量名，如 "model"
    """
    sig = inspect.signature(cls.__init__)
    parameters = sig.parameters

    # 跳过第一个参数（self）
    args_code = []
    for name, param in list(parameters.items())[1:]:  # 跳过 self
        if param.default != inspect.Parameter.empty:
            # 有默认值，生成赋值语句
            value = param.default
            # 如果是表达式（如 8 / 255），保留原始表达式字符串
            if isinstance(value, (int, float)):
                args_code.append(f"{name}={value}")
            else:
                args_code.append(f"{name}={repr(value)}")
        else:
            # 无默认值，使用变量占位符（如 model）
            args_code.append(f"{name}={model_var_name}")

    return f"{instance_name} = {cls.__name__}({', '.join(args_code)})"


def generate():
    for class_name in ATTACKS_FULL:
        cls = globals()[class_name]
        print(f"elif attack_method == '{class_name}':")
        code = generate_class_init_call_code(cls, instance_name='attack', model_var_name='model')
        print('    ', code)

def get_attack(model, attack_method, eps=1/255, alpha=1/255, steps=2):
    # Linf
    if attack_method == 'FGSM':
        attack = FGSM(model=model, eps=eps)
    elif attack_method == 'BIM':
        attack = BIM(model=model, eps=eps, alpha=alpha, steps=steps)
    elif attack_method == 'RFGSM':
        attack = RFGSM(model=model, eps=eps, alpha=alpha, steps=steps)
    elif attack_method == 'PGD':
        attack = PGD(model=model, eps=eps, alpha=alpha, steps=steps, random_start=True)
    elif attack_method == 'ESPGD':
        attack = ESPGD(model=model, eps=eps, alpha=alpha, steps=steps, tau=3, random_start=True)
    elif attack_method == 'EOTPGD':
        attack = EOTPGD(model=model, eps=eps, alpha=alpha, steps=steps, eot_iter=2, random_start=True)
    elif attack_method == 'FFGSM':
        attack = FFGSM(model=model, eps=eps, alpha=alpha)
    elif attack_method == 'TPGD':
        attack = TPGD(model=model, eps=eps, alpha=alpha, steps=steps)
    elif attack_method == 'MIFGSM':
        attack = MIFGSM(model=model, eps=eps, alpha=alpha, steps=steps, decay=1.0)
    elif attack_method == 'UPGD':
        attack = UPGD(model=model, eps=eps, alpha=alpha, steps=steps, random_start=True, loss='ce', decay=1.0, eot_iter=1)
    elif attack_method == 'DIFGSM':
        attack = DIFGSM(model=model, eps=eps, alpha=alpha, steps=steps, decay=1.0, resize_rate=0.9, diversity_prob=0.5, random_start=True)
    elif attack_method == 'TIFGSM':
        attack = TIFGSM(model=model, eps=eps, alpha=alpha, steps=steps, decay=1.0, kernel_name='gaussian', len_kernel=15, nsig=3, resize_rate=0.9, diversity_prob=0.5, random_start=True)
    elif attack_method == 'Jitter':
        attack = Jitter(model=model, eps=eps, alpha=alpha, steps=steps, scale=10, std=0.1, random_start=True)
    elif attack_method == 'NIFGSM':
        attack = NIFGSM(model=model, eps=eps, alpha=alpha, steps=steps, decay=1.0)
    elif attack_method == 'PGDRS':
        attack = PGDRS(model=model, eps=eps, alpha=alpha, steps=steps, noise_type='guassian', noise_sd=0.5, noise_batch_size=5, batch_max=2048)
    elif attack_method == 'SINIFGSM':
        attack = SINIFGSM(model=model, eps=eps, alpha=alpha, steps=steps, decay=1.0, m=5)
    elif attack_method == 'VMIFGSM':
        attack = VMIFGSM(model=model, eps=eps, alpha=alpha, steps=steps, decay=1.0, N=5, beta=1.5)
    elif attack_method == 'VNIFGSM':
        attack = VNIFGSM(model=model, eps=eps, alpha=alpha, steps=steps, decay=1.0, N=5, beta=1.5)
    elif attack_method == 'SPSA':
        attack = SPSA(model=model, eps=eps, delta=eps, lr=0.01, nb_iter=steps, nb_sample=128, max_batch_size=64)
    elif attack_method == 'PIFGSM':
        attack = PIFGSM(model=model, max_epsilon=eps, num_iter_set=steps, momentum=1.0, amplification=10.0, prob=0.7)
    elif attack_method == 'PIFGSMPP':
        attack = PIFGSMPP(model=model, max_epsilon=eps, num_iter_set=steps, momentum=1.0, amplification=10.0, prob=0.7, project_factor=0.8)
    elif attack_method == 'CWLinf':
        attack = CWLinf(model=model, c=1, kappa=0, steps=steps, lr=0.01, abort_early=True)
    elif attack_method == 'CWBSLinf':
        attack = CWBSLinf(model=model, init_c=1, kappa=0, steps=steps, lr=0.01, binary_search_steps=9, abort_early=True)
    elif attack_method == 'FAB':
        attack = FAB(model=model, eps=eps, n_restarts=1, n_iter=steps, alpha_max=0.1, eta=1.05, beta=0.9, las=False)
    elif attack_method == 'ZeroGrad':
        attack = ZeroGrad(model=model, eps=eps, alpha=alpha, qval=0.35, steps=steps, random_start=True)
    
    
    # L0
    elif attack_method == 'JSMA':
        attack = JSMA(model=model, theta=1.0, gamma=0.1, increasing=True)
    elif attack_method == 'CWL0':
        attack = CWL0(model=model, c=1, kappa=0, steps=steps, lr=0.01, abort_early=True)
    elif attack_method == 'CWBSL0':
        attack = CWBSL0(model=model, init_c=1, kappa=0, steps=steps, lr=0.01, binary_search_steps=9, abort_early=True)
    elif attack_method == 'SparseFool':
        attack = SparseFool(model=model, steps=steps, lam=3, overshoot=0.02)
    elif attack_method == 'OnePixel':
        # attack = OnePixel(model=model, pixels=1, steps=steps, popsize=10, inf_batch=128)
        attack = OnePixel(model=model, pixels=10, steps=steps, popsize=10, inf_batch=128)
    elif attack_method == 'Pixle':
        attack = Pixle(model=model, x_dimensions=(2, 10), y_dimensions=(2, 10), pixel_mapping='random', restarts=5, max_iterations=steps, update_each_iteration=False)

    # L1
    elif attack_method == 'EADL1':
        attack = EADL1(model=model, init_c=1, kappa=0, beta=0.001, steps=steps, lr=0.01, binary_search_steps=9, abort_early=True)
    elif attack_method == 'EADEN':  # l1 and l2
        attack = EADEN(model=model, init_c=1, kappa=0, beta=0.001, steps=steps, lr=0.01, binary_search_steps=9, abort_early=True)
    elif attack_method == 'FABL1':
        attack = FABL1(model=model, eps=eps, n_restarts=1, n_iter=steps, alpha_max=0.1, eta=1.05, beta=0.9, las=False)
    
    # L2
    elif attack_method == 'CW':
        attack = CW(model=model, c=1, kappa=0, steps=steps, lr=0.01, abort_early=True)
    elif attack_method == 'CWBS':
        attack = CWBS(model=model, init_c=1, kappa=0, steps=steps, lr=0.01, binary_search_steps=9, abort_early=True)
    elif attack_method == 'PGDL2':
        attack = PGDL2(model=model, eps=2.0, alpha=0.2, steps=steps, random_start=True, eps_for_division=1e-10)
    elif attack_method == 'DeepFool':
        attack = DeepFool(model=model, steps=10, overshoot=0.02)
    elif attack_method == 'PGDRSL2':
        attack = PGDRSL2(model=model, eps=2.0, alpha=0.2, steps=steps, noise_type='guassian', noise_sd=0.5, noise_batch_size=5, batch_max=2048, eps_for_division=1e-10)
    elif attack_method == 'FABL2':
        attack = FABL2(model=model, eps=eps, n_restarts=1, n_iter=10, alpha_max=0.1, eta=1.05, beta=0.9, las=False)
    
    # AutoAttack series: Linf, L2
    elif attack_method == 'AFAB':
        attack = AFAB(model=model, norm='Linf', eps=eps, steps=steps, n_restarts=1, alpha_max=0.1, eta=1.05, beta=0.9, verbose=False, seed=0, multi_targeted=False, n_classes=10)
    elif attack_method == 'APGD':
        attack = APGD(model=model, norm='Linf', eps=eps, steps=steps, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=0.75, verbose=False)
    elif attack_method == 'APGDT':
        attack = APGDT(model=model, norm='Linf', eps=eps, steps=steps, n_restarts=1, seed=0, eot_iter=1, rho=0.75, verbose=False, n_classes=10)
    elif attack_method == 'Square':
        attack = Square(model=model, norm='Linf', eps=eps, n_queries=50, n_restarts=1, p_init=0.8, loss='margin', resc_schedule=True, seed=0, verbose=False)
    elif attack_method == 'AutoAttack':
        attack = AutoAttack(model=model, norm='Linf', eps=eps, version='standard', n_classes=10, seed=None, verbose=False)

    # L0, L1, L2, Linf
    elif attack_method == 'FMN':
        attack = FMN(model=model, norm=float('inf'), steps=steps, alpha_init=1.0, alpha_final=None, gamma_init=0.05, gamma_final=0.001, starting_points=None, binary_search_steps=steps)
    else:
        raise NotImplementedError
    
    return attack

