# https://github.com/rikonaka/adversarial-attacks-pytorch/tree/master
# https://github.com/Harry24k/adversarial-attacks-pytorch

# None attacks
from .attacks.vanila import VANILA
from .attacks.gn import GN

# Linf attacks
from .attacks.fgsm import FGSM
from .attacks.bim import BIM
from .attacks.rfgsm import RFGSM
from .attacks.pgd import PGD
from .attacks.espgd import ESPGD
from .attacks.eotpgd import EOTPGD
from .attacks.ffgsm import FFGSM
from .attacks.tpgd import TPGD
from .attacks.mifgsm import MIFGSM
from .attacks.upgd import UPGD
from .attacks.apgd import APGD
from .attacks.apgdt import APGDT
from .attacks.difgsm import DIFGSM
from .attacks.tifgsm import TIFGSM
from .attacks.jitter import Jitter
from .attacks.nifgsm import NIFGSM
from .attacks.pgdrs import PGDRS
from .attacks.sinifgsm import SINIFGSM
from .attacks.vmifgsm import VMIFGSM
from .attacks.vnifgsm import VNIFGSM
from .attacks.spsa import SPSA
from .attacks.pifgsm import PIFGSM
from .attacks.pifgsmpp import PIFGSMPP
from .attacks.fab import FAB
from .attacks.zerograd import ZeroGrad      # https://github.com/Harry24k/adversarial-attacks-pytorch/pull/204

# L2 attacks
from .attacks.cw import CW
from .attacks.cwl0 import CWL0
from .attacks.cwlinf import CWLinf
from .attacks.cwbs import CWBS
from .attacks.cwbsl0 import CWBSL0
from .attacks.cwbslinf import CWBSLinf
from .attacks.pgdl2 import PGDL2
from .attacks.pgdrsl2 import PGDRSL2
from .attacks.deepfool import DeepFool
from .attacks.eaden import EADEN
from .attacks.fabl2 import FABL2

# L1 attacks
from .attacks.eadl1 import EADL1
from .attacks.fabl1 import FABL1

# L0 attacks
from .attacks.sparsefool import SparseFool
from .attacks.onepixel import OnePixel
from .attacks.pixle import Pixle
from .attacks.jsma import JSMA

# Linf, L2 attacks
from .attacks.autoattack import AutoAttack
from .attacks.square import Square
from .attacks.afab import AFAB

# L0, L1, L2, Linf
from .attacks.fmn import FMN        # https://github.com/Harry24k/adversarial-attacks-pytorch/pull/165

# Wrapper Class
from .wrappers.multiattack import MultiAttack
from .wrappers.lgv import LGV

__version__ = "3.5.1"
__all__ = [
    "VANILA",
    "GN",
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
    "MultiAttack",
    "LGV",
]
__wrapper__ = [
    "LGV",
    "MultiAttack",
]
