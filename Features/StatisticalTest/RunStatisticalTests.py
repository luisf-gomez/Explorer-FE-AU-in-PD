# -*- coding: utf-8 -*-
"""
@author: MSc. Luis Felipe Gómez Gómez - UAM
"""

import numpy as np

Freeze_100 = np.load('./FA_Freeze_100_Original.npy')

Freeze_50 = np.load('./AU_PD_Freeze_50_Original.npy')
Triplet_50 = np.load('./AU_PD_Freeze_50_Triplet.npy')

Freeze_75 = np.load('./AU_PD_Freeze_75_Original.npy')
Triplet_75 = np.load('./AU_PD_Freeze_75_Triplet.npy')

ResNet7 = np.load('./AU_PD_ResNet7_Original.npy')
Triplet_ResNet7 = np.load('./AU_PD_ResNet7_Triplet.npy')

VGG8 = np.load('./AU_PD_VGG8_Original.npy')
Triplet_VGG8 = np.load('./AU_PD_VGG8_Triplet.npy')


alpha_corre = 0.05/3

from scipy.stats import kruskal as f_oneway
from scipy.stats import mannwhitneyu

print('*'*60)
print(f_oneway(Freeze_100, Freeze_75, Triplet_ResNet7))

print('*'*60)
print( mannwhitneyu(Freeze_100, Freeze_75) )
print( mannwhitneyu(Freeze_100, Triplet_ResNet7) )
print( mannwhitneyu(Freeze_75, Triplet_ResNet7) )


