from typing import Tuple, Optional, Union, List
import torch
from ....torchio import DATA
from ....data.subject import Subject
from .. import RandomTransform


class RandomMPRAGE(RandomTransform):
    def __init__(
            self,
            TI: Union[float, Tuple[float, float]] = (600, 1200),
            p: float = 1,
            batch_generation: Optional[int] = None,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.coefficients_range = self.parse_range(
            TI, 'TI_range')
        self.TI = TI
        self.batch_generation = batch_generation

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        TI_list = []
        eps = 1e-6
        if not self.batch_generation:
            for image_name, image_dict in sample.get_images_dict().items():
                TI = torch.FloatTensor(1).uniform_(*self.TI)
                random_parameters_dict = {'TI': TI}
                random_parameters_images_dict[image_name] = random_parameters_dict
                image_dict[DATA] = self.generate_mprage(T1=image_dict[DATA][0, ...]+eps,
                                                        PD=image_dict[DATA][2, ...]+eps,
                                                        TI=TI)
                TI_list.append(TI[0])
            sample['physics'] = torch.FloatTensor(TI_list)
            sample.add_transform(self, random_parameters_images_dict)
        else:
            for image_name, image_dict in sample.get_images_dict().items():
                batch_vols = torch.zeros((self.batch_generation,) + image_dict[DATA].shape[1:])
                for generation in range(self.batch_generation):
                    TI = torch.FloatTensor(1).uniform_(*self.TI)
                    batch_vols[generation, ...] = self.generate_mprage(T1=image_dict[DATA][0, ...] + eps,
                                                                       PD=image_dict[DATA][2, ...] + eps,
                                                                       TI=TI)
                    TI_list.append(TI[0])
                random_parameters_dict = {'TI': TI}
                random_parameters_images_dict[image_name] = random_parameters_dict
                sample['physics'] = torch.FloatTensor(TI_list)
                sample.add_transform(self, random_parameters_images_dict)
            image_dict[DATA] = batch_vols  # [:, None, ...]
        return sample

    @staticmethod
    def generate_mprage(T1, PD, TI, TD=10e-3, tau=10e-3, Gs=1):
        mprage_img = Gs * PD * (1 - 2 * torch.exp(-TI / T1) / (1 + torch.exp(-(TI + TD + tau) / T1)))
        # print(mprage_img[40:80, 40:80, 40:80])
        # print(f'The min and max of the images is {mprage_img.squeeze().detach().cpu().numpy().max()},'
        #       f'{mprage_img.squeeze().detach().cpu().numpy().max()}')
        mprage_img[torch.isnan(mprage_img)] = 0.0
        mprage_img[torch.isinf(mprage_img)] = 0.0
        mprage_img[mprage_img < 0] = 0.0
        # mprage_img[torch.abs(mprage_img) > 1] = 0.0
        # topk_vals, _ = torch.topk(torch.flatten(mprage_img), 100)
        # mprage_img[mprage_img > topk_vals[-1]] = topk_vals[-1]
        return mprage_img[None, ...]
