import io

import numpy as np
import pytest

from speech2motion.data_structures.restpose import Restpose


@pytest.fixture
def restpose():
    """Fixture to create a Restpose instance for testing."""
    name = "test_restpose"
    joint_names = ["joint1", "joint2", "joint3"]
    local_matrices = np.random.rand(len(joint_names), 4, 4)
    matrix_world = np.random.rand(4, 4)
    parent_indices = [-1, 0, 1]
    # joint1 is root, joint2 is child of joint1, joint3 is child of joint2
    return Restpose(
        name=name,
        joint_names=joint_names,
        local_matrices=local_matrices,
        matrix_world=matrix_world,
        parent_indices=parent_indices
    )


def test_init(restpose):
    """Test the initialization of Restpose."""
    assert restpose.name == "test_restpose"
    assert restpose.joint_names == ["joint1", "joint2", "joint3"]
    assert restpose.local_matrices.shape == (3, 4, 4)
    assert restpose.matrix_world.shape == (4, 4)
    assert restpose.parent_indices == [-1, 0, 1]
    assert restpose.n_joints == 3
    with pytest.raises(ValueError):
        Restpose(
            name='WrongLocalMatrices',
            joint_names=["joint1", "joint2", "joint3"],
            local_matrices=np.random.rand(2, 4, 4),
            matrix_world=np.random.rand(4, 4),
            parent_indices=[-1, 0, 1]
        )
    with pytest.raises(ValueError):
        Restpose(
            name='WrongParentIndices',
            joint_names=["joint1", "joint2", "joint3"],
            local_matrices=np.random.rand(3, 4, 4),
            matrix_world=np.random.rand(4, 4),
            parent_indices=[-1, 0]
        )

def test_get_joint_index(restpose):
    """Test getting joint index from Restpose."""
    assert restpose.get_joint_index("joint1") == 0
    assert restpose.get_joint_index("joint2") == 1
    assert restpose.get_joint_index("joint3") == 2
    with pytest.raises(ValueError):
        restpose.get_joint_index("joint4")

def test_to_dict(restpose):
    """Test converting Restpose to dictionary."""
    pose_dict = restpose.to_dict()
    assert pose_dict["name"] == restpose.name
    assert pose_dict["joint_names"] == restpose.joint_names
    assert np.array_equal(pose_dict["local_matrices"], restpose.local_matrices)
    assert np.array_equal(pose_dict["matrix_world"], restpose.matrix_world)
    assert pose_dict["parent_indices"] == restpose.parent_indices


def test_to_npz(restpose):
    """Test converting Restpose to npz."""
    npz_file = restpose.to_npz()
    assert npz_file is not None


def test_from_dict():
    """Test constructing Restpose from dictionary."""
    restpose_dict = {
        "name": "test_pose",
        "joint_names": ["joint1", "joint2", "joint3"],
        "local_matrices": np.random.rand(3, 4, 4),
        "matrix_world": np.random.rand(4, 4),
        "parent_indices": [-1, 0, 1]
    }
    pose = Restpose.from_dict(restpose_dict)
    assert pose.name == "test_pose"
    assert pose.n_joints == 3
    assert np.array_equal(pose.local_matrices, restpose_dict["local_matrices"])
    assert np.array_equal(pose.matrix_world, restpose_dict["matrix_world"])
    assert pose.parent_indices == restpose_dict["parent_indices"]


def test_from_npz():
    """Test constructing Restpose from npz."""
    restpose_dict = {
        "name": "test_pose",
        "joint_names": ["joint1", "joint2", "joint3"],
        "local_matrices": np.random.rand(3, 4, 4),
        "matrix_world": np.random.rand(4, 4),
        "parent_indices": [-1, 0, 1]
    }
    npz_io = io.BytesIO()
    np.savez_compressed(npz_io, **restpose_dict)
    npz_io.seek(0)
    pose = Restpose.from_npz(npz_io)
    assert pose.name == "test_pose"
    assert pose.n_joints == 3


def test_from_npz_with_parents():
    """Test constructing Restpose from npz with
    parents field instead of parent_indices."""
    restpose_dict = {
        "name": "test_pose",
        "joint_names": ["joint1", "joint2", "joint3"],
        "local_matrices": np.random.rand(3, 4, 4),
        "matrix_world": np.random.rand(4, 4),
        "parents": ["", "joint1", "joint2"]
    }
    npz_io = io.BytesIO()
    np.savez_compressed(npz_io, **restpose_dict)
    npz_io.seek(0)
    pose = Restpose.from_npz(npz_io)
    assert pose.name == "test_pose"
    assert pose.n_joints == 3
    assert pose.parent_indices == [-1, 0, 1]  # Should be converted correctly


def test_dump_and_load_dict(restpose):
    """Test dumping and loading Restpose via dict."""
    pose_dict = restpose.to_dict()
    reloaded_pose = Restpose.from_dict(pose_dict)
    assert reloaded_pose.name == restpose.name
    assert reloaded_pose.joint_names == restpose.joint_names
    assert np.array_equal(reloaded_pose.local_matrices, restpose.local_matrices)
    assert np.array_equal(reloaded_pose.matrix_world, restpose.matrix_world)
    assert reloaded_pose.parent_indices == restpose.parent_indices


def test_dump_and_load_npz(restpose):
    """Test dumping and loading Restpose via npz."""
    npz_io = restpose.to_npz()
    reloaded_pose = Restpose.from_npz(
        npz_io, float_dtype=restpose.local_matrices.dtype)
    assert reloaded_pose.name == restpose.name
    assert reloaded_pose.joint_names == restpose.joint_names
    assert np.array_equal(reloaded_pose.local_matrices, restpose.local_matrices)
    assert np.array_equal(reloaded_pose.matrix_world, restpose.matrix_world)
    assert (reloaded_pose.parent_indices == restpose.parent_indices).all()
