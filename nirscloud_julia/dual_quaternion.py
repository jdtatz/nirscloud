import numpy as np
from numpy.typing import ArrayLike
import quaternion
from scipy.spatial.transform import Rotation, RotationSpline
import scipy.interpolate as interpolate
from typing import Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum, auto

__all__ = ["DualQuaternion"]


@dataclass
class DualQuaternion:
    """ℍ𝔻 = ℍ⊗𝔻 ~ 𝔻⊗ℍ"""

    # real: npt.NDArray[np.quaternion]
    # dual: npt.NDArray[np.quaternion]
    real: np.ndarray
    dual: np.ndarray

    def __postinit__(self):
        assert np.dtype(self.real) == np.quaternion
        assert np.dtype(self.dual) == np.quaternion
        assert np.shape(self.real) == np.shape(self.dual)

    @classmethod
    def identity(cls) -> "DualQuaternion":
        return DualQuaternion(real=quaternion.one, dual=quaternion.zero)

    @classmethod
    def from_rigid_position(
        cls, position: ArrayLike, orientation: np.ndarray
    ) -> "DualQuaternion":
        # TODO orientation: npt.NDArray[np.quaternion]
        real = orientation
        return DualQuaternion(
            real=real,
            dual=0.5 * quaternion.from_vector_part(position) * real,
        )

    @classmethod
    def from_rigid_transform(
        cls, translation: ArrayLike, rotation: Rotation
    ) -> "DualQuaternion":
        # scipy.spatial.transform.Rotation is scalar-last,and numpy-quaternion is scalar-first
        real = quaternion.as_quat_array(rotation.as_quat()[..., [3, 0, 1, 2]])
        return DualQuaternion(
            real=real,
            dual=0.5 * quaternion.from_vector_part(translation) * real,
        )

    @classmethod
    def from_3d_vector(
        cls, vector: ArrayLike, vector_axis: int = -1
    ) -> "DualQuaternion":
        assert np.shape(vector)[vector_axis] == 3
        dual = quaternion.from_vector_part(vector, vector_axis=vector_axis)
        return DualQuaternion(
            real=np.broadcast_to(quaternion.one, dual.shape),
            dual=dual,
        )

    def __add__(self, rhs: "DualQuaternion") -> "DualQuaternion":
        if isinstance(rhs, DualQuaternion):
            return DualQuaternion(real=self.real + rhs.real, dual=self.dual + rhs.dual)
        return NotImplemented

    def __mul__(self, rhs: Union[ArrayLike, "DualQuaternion"]) -> "DualQuaternion":
        if isinstance(rhs, DualQuaternion):
            return DualQuaternion(
                real=self.real * rhs.real,
                dual=self.real * rhs.dual + self.dual * rhs.real,
            )
        else:
            scale = np.broadcast_to(np.asarray(rhs), self.real.shape)
            return DualQuaternion(real=self.real * scale, dual=self.dual * scale)

    def __rmul__(self, lhs: ArrayLike) -> "DualQuaternion":
        scale = np.broadcast_to(np.asarray(lhs), self.real.shape)
        return DualQuaternion(real=self.real * scale, dual=self.dual * scale)

    def shape(self) -> Tuple[int, ...]:
        return np.shape(self.real)

    def quaternion_conjugate(self) -> "DualQuaternion":
        return DualQuaternion(
            real=np.conjugate(self.real), dual=np.conjugate(self.dual)
        )

    def dual_conjugate(self) -> "DualQuaternion":
        return DualQuaternion(real=self.real, dual=-self.dual)

    def combined_conjugate(self) -> "DualQuaternion":
        return self.quaternion_conjugate().dual_conjugate()

    def inverse(self) -> "DualQuaternion":
        p_ = 1 / self.real
        return DualQuaternion(real=p_, dual=p_ * (-self.dual * p_))

    def rotational_componet(self) -> Rotation:
        return Rotation.from_quat(
            quaternion.as_float_array(self.real)[..., [1, 2, 3, 0]]
        )

    def translational_componet(self) -> np.ndarray:
        return quaternion.as_vector_part(2 * self.dual * np.conjugate(self.real))

    def apply(self, v: ArrayLike, vector_axis: int = -1) -> np.ndarray:
        q = DualQuaternion.from_3d_vector(v, vector_axis=vector_axis)
        qp = self * q * self.combined_conjugate()
        return quaternion.as_vector_part(qp.dual)


class RigidMotionInterpolationMethod(Enum):
    Split = auto()
    Joint = auto()


class RigidMotionInterpolation:
    time: np.ndarray
    dq: DualQuaternion
    method: RigidMotionInterpolationMethod
    interpolater: Any

    def __init__(
        self,
        time: ArrayLike,
        dq: DualQuaternion,
        method: RigidMotionInterpolationMethod = RigidMotionInterpolationMethod.Split,
    ):
        time = np.asanyarray(time)
        assert np.ndim(time) == 1
        assert np.shape(time) == dq.shape()
        self.time = time
        self.dq = dq
        self.method = method
        if method is RigidMotionInterpolationMethod.Split:
            (t, c, k), *_ = interpolate.splprep(dq.translational_componet(), u=time)
            path_interpolater = interpolate.BSpline(t, c, k, axis=1)
            rot_interpolater = RotationSpline(time, dq.rotational_componet())
            self.interpolater = path_interpolater, rot_interpolater
        elif method is RigidMotionInterpolationMethod.Joint:
            raise NotImplementedError
        else:
            raise TypeError(
                f"method: {method} is not a member of RigidMotionInterpolationMethod"
            )

    def __call__(self, t: ArrayLike) -> DualQuaternion:
        if self.method is RigidMotionInterpolationMethod.Split:
            path_interpolater, rot_interpolater = self.interpolater
            path = path_interpolater(t)
            rot = rot_interpolater(t)
            return DualQuaternion.from_rigid_transform(path, rot)
        elif self.method is RigidMotionInterpolationMethod.Joint:
            raise NotImplementedError
        else:
            raise TypeError(
                f"Somehow self.method: {self.method} is not a member of RigidMotionInterpolationMethod"
            )
