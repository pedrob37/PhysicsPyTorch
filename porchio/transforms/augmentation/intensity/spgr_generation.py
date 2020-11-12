from typing import Tuple, Optional, Union, List
import torch
from ....torchio import DATA
from ....data.subject import Subject
from .. import RandomTransform
import numpy as np


class RandomSPGR(RandomTransform):
    def __init__(
            self,
            TR: Union[float, Tuple[float, float]] = (0.005, 2.0),
            TE: Union[float, Tuple[float, float]] = (0.005, 0.1),
            FA: Union[float, Tuple[float, float]] = (5.0, 90.0),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.coefficients_range_TR = self.parse_range(
            TR, 'TR_range')
        self.TR = TR
        self.coefficients_range_TE = self.parse_range(
            TE, 'TE_range')
        self.TE = TE
        self.coefficients_range_FA = self.parse_range(
            FA, 'FA_range')
        self.FA = FA

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        TR_list = []
        TE_list = []
        FA_list = []
        eps = 1e-6
        for image_name, image_dict in sample.get_images_dict().items():
            TR = torch.FloatTensor(1).uniform_(*self.TR)
            TE = torch.FloatTensor(1).uniform_(self.TE[0], min(TR.numpy()[0], self.TE[1]))
            FA = torch.FloatTensor(1).uniform_(*self.FA)
            random_parameters_dict = {'TR': TR,
                                      'TE': TE,
                                      'FA': FA}
            random_parameters_images_dict[image_name] = random_parameters_dict
            image_dict[DATA] = self.generate_spgr(T1=image_dict[DATA][0, ...]+eps,
                                                  T2s=image_dict[DATA][1, ...]+eps,
                                                  PD=image_dict[DATA][2, ...]+eps,
                                                  TR=TR, TE=TE, FA=FA)
            # raise RuntimeWarning(f'The spatial shape is {image_dict[DATA].shape}')
            TR_list.append(TR[0])
            TE_list.append(TE[0])
            FA_list.append(FA[0])
        sample['physics'] = torch.stack([torch.FloatTensor(TR_list),
                                         torch.FloatTensor(TE_list),
                                         torch.FloatTensor(FA_list)], dim=1)
        sample.add_transform(self, random_parameters_images_dict)
        return sample

    @staticmethod
    def generate_spgr(T1, T2s, PD, TR, TE, FA, Gs=1):
        FA = FA * np.pi / 180
        spgr_img = Gs * PD * torch.sin(FA) * (1 - torch.exp(-TR / T1)) * torch.exp(-TE / T2s) / (1 - torch.cos(FA) * torch.exp(-TR / T1))
        spgr_img[torch.isnan(spgr_img)] = 0.0
        spgr_img[torch.isinf(spgr_img)] = 0.0
        spgr_img[spgr_img < 0] = 0.0
        spgr_img[torch.abs(spgr_img) > 1] = 0.0
        topk_vals, _ = torch.topk(torch.flatten(spgr_img), 100)
        spgr_img[spgr_img > topk_vals[-1]] = topk_vals[-1]
        return spgr_img[None, ...]
