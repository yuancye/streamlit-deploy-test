import inspect
import numpy as np
import pprint
from typing import Any, List, Optional, Tuple
from fvcore.transforms.transform import Transform, TransformList

"""
See "Data Augmentation" tutorial for an overview of the system:
https://detectron2.readthedocs.io/tutorials/augmentation.html
"""


__all__ = [
    "Augmentation"
]


def _get_aug_input_args(aug, aug_input) -> List[Any]:
    """
    Get the arguments to be passed to ``aug.get_transform`` from the input ``aug_input``.
    """
    if aug.input_args is None:
        # Decide what attributes are needed automatically
        prms = list(inspect.signature(aug.get_transform).parameters.items())
        # The default behavior is: if there is one parameter, then its "image"
        # (work automatically for majority of use cases, and also avoid BC breaking),
        # Otherwise, use the argument names.
        if len(prms) == 1:
            names = ("image",)
        else:
            names = []
            for name, prm in prms:
                if prm.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    raise TypeError(
                        f""" \
The default implementation of `{type(aug)}.__call__` does not allow \
`{type(aug)}.get_transform` to use variable-length arguments (*args, **kwargs)! \
If arguments are unknown, reimplement `__call__` instead. \
"""
                    )
                names.append(name)
        aug.input_args = tuple(names)

    args = []
    for f in aug.input_args:
        try:
            args.append(getattr(aug_input, f))
        except AttributeError as e:
            raise AttributeError(
                f"{type(aug)}.get_transform needs input attribute '{f}', "
                f"but it is not an attribute of {type(aug_input)}!"
            ) from e
    return args


class Augmentation:
    """
    Augmentation defines (often random) policies/strategies to generate :class:`Transform`
    from data. It is often used for pre-processing of input data.

    A "policy" that generates a :class:`Transform` may, in the most general case,
    need arbitrary information from input data in order to determine what transforms
    to apply. Therefore, each :class:`Augmentation` instance defines the arguments
    needed by its :meth:`get_transform` method. When called with the positional arguments,
    the :meth:`get_transform` method executes the policy.

    Note that :class:`Augmentation` defines the policies to create a :class:`Transform`,
    but not how to execute the actual transform operations to those data.
    Its :meth:`__call__` method will use :meth:`AugInput.transform` to execute the transform.

    The returned `Transform` object is meant to describe deterministic transformation, which means
    it can be re-applied on associated data, e.g. the geometry of an image and its segmentation
    masks need to be transformed together.
    (If such re-application is not needed, then determinism is not a crucial requirement.)
    """

    input_args: Optional[Tuple[str]] = None
    """
    Stores the attribute names needed by :meth:`get_transform`, e.g.  ``("image", "sem_seg")``.
    By default, it is just a tuple of argument names in :meth:`self.get_transform`, which often only
    contain "image". As long as the argument name convention is followed, there is no need for
    users to touch this attribute.
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def get_transform(self, *args) -> Transform:
        """
        Execute the policy based on input data, and decide what transform to apply to inputs.

        Args:
            args: Any fixed-length positional arguments. By default, the name of the arguments
                should exist in the :class:`AugInput` to be used.

        Returns:
            Transform: Returns the deterministic transform to apply to the input.

        Examples:
        ::
            class MyAug:
                # if a policy needs to know both image and semantic segmentation
                def get_transform(image, sem_seg) -> T.Transform:
                    pass
            tfm: Transform = MyAug().get_transform(image, sem_seg)
            new_image = tfm.apply_image(image)

        Notes:
            Users can freely use arbitrary new argument names in custom
            :meth:`get_transform` method, as long as they are available in the
            input data. In detectron2 we use the following convention:

            * image: (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
              floating point in range [0, 1] or [0, 255].
            * boxes: (N,4) ndarray of float32. It represents the instance bounding boxes
              of N instances. Each is in XYXY format in unit of absolute coordinates.
            * sem_seg: (H,W) ndarray of type uint8. Each element is an integer label of pixel.

            We do not specify convention for other types and do not include builtin
            :class:`Augmentation` that uses other types in detectron2.
        """
        raise NotImplementedError

    def __call__(self, aug_input) -> Transform:
        """
        Augment the given `aug_input` **in-place**, and return the transform that's used.

        This method will be called to apply the augmentation. In most augmentation, it
        is enough to use the default implementation, which calls :meth:`get_transform`
        using the inputs. But a subclass can overwrite it to have more complicated logic.

        Args:
            aug_input (AugInput): an object that has attributes needed by this augmentation
                (defined by ``self.get_transform``). Its ``transform`` method will be called
                to in-place transform it.

        Returns:
            Transform: the transform that is applied on the input.
        """
        args = _get_aug_input_args(self, aug_input)
        tfm = self.get_transform(*args)
        assert isinstance(tfm, (Transform, TransformList)), (
            f"{type(self)}.get_transform must return an instance of Transform! "
            f"Got {type(tfm)} instead."
        )
        aug_input.transform(tfm)
        return tfm

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyAugmentation(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(name)
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                attr_str = pprint.pformat(attr)
                if "\n" in attr_str:
                    # don't show it if pformat decides to use >1 lines
                    attr_str = "..."
                argstr.append("{}={}".format(name, attr_str))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__

