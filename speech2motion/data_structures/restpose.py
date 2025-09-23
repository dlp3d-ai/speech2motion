import io
from typing import Any

import numpy as np

from ..utils.io import load_npz
from ..utils.log import setup_logger
from ..utils.super import Super


class Restpose(Super):
    """Class for storing and managing armature rest pose data.

    Rest pose is the default or base pose of an armature, defining the original
    state of bones before any animation transformations are applied. It serves
    as the reference frame for all animations and deformations, ensuring proper
    alignment between bones and mesh vertices.

    This class stores the hierarchical structure of joints, their local
    transformation matrices, and the world transformation matrix that defines
    the armature's position and orientation in world space.
    """

    def __init__(self,
                 name: str,
                 joint_names: list[str],
                 local_matrices: np.ndarray,
                 matrix_world: np.ndarray,
                 parent_indices: list[int],
                 logger_cfg: None | dict = None) -> None:
        """Initialize restpose object.

        Args:
            name (str):
                Restpose name.
            joint_names (list[str]):
                List of joint names.
            local_matrices (np.ndarray):
                Transformation matrices between
                joint coordinate system and armature space
                with shape (n_joints, 4, 4).
            matrix_world (np.ndarray):
                Transformation matrix between world coordinate system and armature space
                with shape (4, 4).
            parent_indices (list[int]):
                List of parent joint indices for each joint, length is n_joints.
            logger_cfg (None | dict[str, Any], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Defaults to None.
        """
        Super.__init__(self, logger_cfg)
        self.name = name
        # check n_joints
        if local_matrices.shape[0] != len(joint_names):
            msg = f'In Restpose {name}, ' +\
                f'number of joints in local_matrices {local_matrices.shape[0]} ' +\
                f'does not match length of joint_names {len(joint_names)}.'
            self.logger.error(msg)
            raise ValueError(msg)
        if len(parent_indices) != len(joint_names):
            msg = f'In Restpose {name}, ' +\
                f'length of parent_indices {len(parent_indices)} ' +\
                f'does not match length of joint_names {len(joint_names)}.'
            self.logger.error(msg)
            raise ValueError(msg)
        self.joint_names = joint_names
        self.local_matrices = local_matrices
        self.matrix_world = matrix_world
        self.parent_indices = parent_indices
        self.joint_name_index_mapping = {
            joint_name: joint_idx
            for joint_idx, joint_name in enumerate(self.joint_names)}

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def get_joint_index(self, joint_name: str) -> int:
        """Get index corresponding to joint name.

        Args:
            joint_name (str): Joint name.

        Returns:
            int: Index corresponding to joint name.
        """
        if joint_name not in self.joint_name_index_mapping:
            msg = f'Joint name {joint_name} is not in Restpose named {self.name}.'
            self.logger.error(msg)
            raise ValueError(msg)
        return self.joint_name_index_mapping[joint_name]

    def to_dict(self) -> dict[str, Any]:
        """Convert current Restpose to dictionary.

        Returns:
            dict: Dictionary representation of current Restpose.
        """
        ret_dict = dict(
            name=self.name,
            joint_names=self.joint_names,
            local_matrices=self.local_matrices,
            matrix_world=self.matrix_world,
            parent_indices=self.parent_indices,
        )
        return ret_dict

    def to_npz(self) -> io.BytesIO:
        """Convert current Restpose to compressed npz file.

        Returns:
            io.BytesIO: Compressed npz file.
        """
        npz_dict = self.to_dict()
        ret_io = io.BytesIO()
        np.savez_compressed(ret_io, **npz_dict)
        ret_io.seek(0)
        return ret_io

    @classmethod
    def from_dict(cls,
                  restpose_dict: dict[str, Any],
                  logger_cfg: None | dict[str, Any] = None) -> 'Restpose':
        """Construct Restpose from dictionary.

        Args:
            restpose_dict (dict): Dictionary representation of Restpose.
            logger_cfg (Union[None, dict], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Defaults to None.

        Returns:
            Restpose: Restpose constructed from dictionary.
        """
        restpose_name = restpose_dict['name']
        joint_names = restpose_dict['joint_names']
        local_matrices = restpose_dict['local_matrices']
        matrix_world = restpose_dict['matrix_world']
        parent_indices = restpose_dict['parent_indices']
        return Restpose(
            name=restpose_name,
            joint_names=joint_names,
            local_matrices=local_matrices,
            matrix_world=matrix_world,
            parent_indices=parent_indices,
            logger_cfg=logger_cfg)

    @classmethod
    def from_npz(cls,
                 npz_io: io.BytesIO,
                 name_override: None | str = None,
                 float_dtype: np.floating = np.float32,
                 logger_cfg: None | dict[str, Any] = None) -> 'Restpose':
        """Construct Restpose from compressed npz file.

        Args:
            npz_io (io.BytesIO): Compressed npz file.
            name_override (Union[None, str], optional):
                If provided, use this name instead of 'name' key in npz_dict.
                Defaults to None.
            float_dtype (np.floating):
                Data type of numpy floating point arrays in returned Restpose.
                Defaults to np.float32. Please avoid using np.float16 as np.linalg
                does not support np.float16.
            logger_cfg (Union[None, dict], optional):
                Logger configuration, see `setup_logger` for detailed description.
                Defaults to None.

        Returns:
            Restpose: Restpose constructed from compressed npz file.
        """
        npz_dict = load_npz(npz_io, float_dtype)
        # if parent_indices is not in npz_dict,
        # try to read parents in list[str] format and build indices
        if 'parent_indices' not in npz_dict:
            if 'parents' in npz_dict:
                parents = npz_dict['parents']
                joint_name_index_mapping = {
                    joint_name: joint_idx
                    for joint_idx, joint_name in enumerate(npz_dict['joint_names'])}
                parent_indices = list()
                for parent_name in parents:
                    if parent_name not in joint_name_index_mapping:
                        parent_indices.append(-1)
                    else:
                        parent_indices.append(joint_name_index_mapping[parent_name])
                npz_dict['parent_indices'] = parent_indices
            else:
                msg = ('npz_dict has neither parent_indices nor parents, '
                       'cannot construct Restpose.')
                if logger_cfg is None:
                    logger_cfg = dict(logger_name=cls.__name__)
                else:
                    logger_cfg = logger_cfg.copy()
                    logger_cfg["logger_name"] = cls.__name__
                logger = setup_logger(logger_cfg)
                logger.error(msg)
                raise ValueError(msg)
        if name_override is not None:
            npz_dict['name'] = name_override
        return cls.from_dict(npz_dict, logger_cfg=logger_cfg)
