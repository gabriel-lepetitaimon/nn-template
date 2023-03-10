import torch
from functools import cached_property
from ..config import Cfg


class TestTimeAugmentCfg(Cfg.Obj):
    alias = Cfg.str(default=None)

    flip = Cfg.oneOf(False, True, 'horizontal', 'vertical', default=False)
    rot90 = Cfg.str(None)
    merge_mode = Cfg.oneOf('mean', 'gmean', 'sum', 'max', 'min', 'tsharpen', default='mean')

    @alias.check_value
    def check_alias(self, value: str | None):
        if value is None:
            return None

        import ttach as tta
        value = value.strip().lower()
        if not value.endswith('transform'):
            value += '_transform'
        transform = getattr(tta.aliases, value, None)
        if transform is None or \
           not isinstance(transform(), (tta.base.BaseTransform, tta.base.Compose, tta.base.Chain)):
            raise Cfg.InvalidAttr(f'"{value}" is an invalid ttach alias for attribute {self.fullname}')
        return value

    @rot90.check_value
    def check_rot90(self, rot90: str | None):
        if rot90 is None:
            return None
        if rot90 == 'all':
            return [0, 90, 180, 270]
        rot90 = rot90.split(',')

        error = Cfg.InvalidAttr(f"{rot90} is not a valid angle for attribute {self.fullname+'.rot90'}",
                                f"Must be one of [0, 90, 180, 270] or a list of those.")
        try:
            rot90 = [Cfg.int.interpret(_, False) for _ in rot90 if _]
        except TypeError:
            raise error from None
        if not all(_ in (0, 90, 180, 270) for _ in rot90):
            raise error
        return rot90

    @alias.post_checker
    def check_alias(self, alias):
        import ttach as tta
        if not hasattr(tta.aliases, alias):
            if not alias.endswith('_transform') and hasattr(tta.aliases, alias+'_transform'):
                return alias+'_transform'
            else:
                raise Cfg.InvalidAttr(f'Invalid test time augmentation alias "{alias}"')

    @cached_property
    def transforms(self):
        import ttach as tta
        if self.alias:
            return getattr(tta.aliases, self.alias)()

        transforms = []

        match self.flip:
            case True:
                transforms += [tta.HorizontalFlip(), tta.VerticalFlip()]
            case 'horizontal':
                transforms += [tta.HorizontalFlip()]
            case 'vertical':
                transforms += [tta.VerticalFlip()]

        if self.rot90:
            transforms += [tta.Rotate90(angles=self.rot90)]

        return tta.Compose(transforms)

    def create_merger(self):
        import ttach as tta
        return tta.base.Merger(type=self.merge_mode, n=len(self.transforms))

    def configure_test_time_augment(self, model):
        import ttach as tta
        return tta.SegmentationTTAWrapper(model, transforms=self.transforms)
