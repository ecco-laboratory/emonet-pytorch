# %%
# imports
from typing import Any, Callable, List, Optional, Tuple

import torchvision
from torchvision.datasets import VisionDataset
# %%
# Hot and sexy torch Dataset class for Alan's videos

class Cowen2017Dataset(VisionDataset):
    """`Cowen & Keltner (2017) <https://www.pnas.org/doi/full/10.1073/pnas.1702247114>` PyTorch-style Dataset.

    This dataset returns each video as a 4D tensor.

    Args:
        root (string): Enclosing folder where videos are located on the local machine.
        annFile (string): Path to directory of metadata/annotation CSVs.
        censorFile (boolean, optional): Censor Alan's "bad" videos? Defaults to True.
        train (boolean, optional): If True, creates dataset from Kragel et al. (2019)'s training set, otherwise
            from the testing set. Defaults to True.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root: str,
                 annPath: str,
                 censor: bool = True,
                 classes: str = None,
                 train: bool = True,
                 device: str = 'cpu',
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)

        import os

        import pandas as pd

        self.train = train
        self.device = device

        # Read in the Cowen & Keltner top-1 "winning" human emotion classes
        self.labels = pd.read_csv(os.path.join(annPath, f"kragel2019_{'train' if self.train else 'test'}_video_ids.csv"),
                                   index_col='video')
        self.labels = self.labels[self.labels['emotion'].isin(emonet_output_classes)]
        
        # Flexibly subsample emotion classes based on user input
        if classes is not None:
            self.labels = self.labels[self.labels['emotion'].isin(classes)]

        if censor:
            # Truly I wish this was in long form but Alan doesn't like tidy data does he
            censored = pd.read_csv(os.path.join(annPath, 'cowen2017_censored_video_ids.csv'))
            # We don't need to see the censored ones! At least I personally don't
            # I guess the model doesn't have feelings
            self.labels = self.labels[~self.labels.index.isin(censored['less.bad'])]
            self.labels = self.labels[~self.labels.index.isin(censored['very.bad'])]

        self.ids = self.labels.index.to_list()
    
    def _load_video(self, id: str):
        import os

        video = torchvision.io.read_video(os.path.join(self.root, self.labels.loc[id]['emotion'], id),
                                          pts_unit='sec')
        # None of the videos have audio, so discard that from the loaded tuple
        # Also for convenience, discard dict labeling fps so that the videos look like 4D imgs
        # with dims frames x channels x height x width ... which is NOT the default order!
        frames = video[0].permute((0, 3, 1, 2))

        # From when I was still trying to output a read_video item
        # video = (frames, video[1], video[2])

        return frames
    
    def _load_target(self, id: str) -> List[Any]:
        target = self.labels.loc[id].to_dict()
        target['id'] = id
        
        return target
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        video = self._load_video(id)
        target = self._load_target(id)

        if self.transforms is not None:
            video, target = self.transforms(video, target)
            
        video.to(device=self.device)

        if self.target_transform is not None:
            target.to(device=self.device)

        return video, target

    def __len__(self) -> int:
        return len(self.ids)


# %%
# Phil's 20 EmoNet output classes as global variable
emonet_output_classes = [
    'Adoration',
    'Aesthetic Appreciation',
    'Amusement',
    'Anxiety',
    'Awe',
    'Boredom',
    'Confusion',
    'Craving',
    'Disgust',
    'Empathic Pain',
    'Entrancement',
    'Excitement',
    'Fear',
    'Horror',
    'Interest',
    'Joy',
    'Romance',
    'Sadness',
    'Sexual Desire',
    'Surprise'
]
