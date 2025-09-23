import io

import numpy as np
import pytest

from speech2motion.data_structures.motion_clip import MotionClip


@pytest.fixture
def motion_clip():
    """Fixture to create a MotionClip instance for testing."""
    n_frames = 10
    joint_names = ['joint1', 'joint2']
    joint_rotmat = np.random.rand(n_frames, len(joint_names), 3, 3)
    root_world_position = np.random.rand(n_frames, 3)
    return MotionClip(
        n_frames=n_frames,
        joint_names=joint_names,
        joint_rotmat=joint_rotmat,
        root_world_position=root_world_position,
        restpose_name='restpose',
        app_name='python_backend',
        cutoff_frames=[(0, 1, 1), (n_frames - 1, 1, 1)],
        cutoff_ranges=[(0, 5, 1, 1), (6, 10, 1, 1)]
    )

@pytest.fixture
def motion_clip_with_blendshapes():
    """Fixture to create a MotionClip instance with blendshapes for testing."""
    n_frames = 10
    joint_names = ['joint1', 'joint2']
    joint_rotmat = np.random.rand(n_frames, len(joint_names), 3, 3)
    root_world_position = np.random.rand(n_frames, 3)
    blendshape_names = ['blendshape1', 'blendshape2', 'blendshape3']
    blendshape_values = np.random.rand(n_frames, len(blendshape_names))
    return MotionClip(
        n_frames=n_frames,
        joint_names=joint_names,
        joint_rotmat=joint_rotmat,
        root_world_position=root_world_position,
        restpose_name='restpose',
        app_name='python_backend',
        cutoff_frames=[(0, 1, 1), (n_frames - 1, 1, 1)],
        cutoff_ranges=[(0, 5, 1, 1), (6, 10, 1, 1)],
        blendshape_names=blendshape_names,
        blendshape_values=blendshape_values
    )

def test_init(motion_clip):
    """Test the initialization of MotionClip."""
    assert motion_clip.n_frames == 10
    assert motion_clip.joint_names == ['joint1', 'joint2']
    assert motion_clip.joint_rotmat.shape == (10, 2, 3, 3)
    assert motion_clip.root_world_position.shape == (10, 3)
    assert motion_clip.app_name == 'python_backend'

def test_set_joint_rotmat(motion_clip):
    """Test setting joint rotation matrix."""
    new_joint_rotmat = np.random.rand(10, 2, 3, 3)
    motion_clip.set_joint_rotmat(new_joint_rotmat, ['joint1', 'joint2'])
    assert np.array_equal(motion_clip.joint_rotmat, new_joint_rotmat)

def test_set_cutoff_frames(motion_clip):
    """Test setting cutoff frames."""
    cutoff_frames = [(0, 1, 2), (5, 1, 2)]
    motion_clip.set_cutoff_frames(cutoff_frames)
    assert motion_clip.cutoff_frames == cutoff_frames

def test_set_cutoff_ranges(motion_clip):
    """Test setting cutoff ranges."""
    cutoff_ranges = [(0, 5, 1, 2), (5, 10, 1, 2)]
    motion_clip.set_cutoff_ranges(cutoff_ranges)
    assert motion_clip.cutoff_ranges == cutoff_ranges

def test_clone(motion_clip):
    """Test cloning a MotionClip."""
    cloned_clip = motion_clip.clone()
    assert cloned_clip.n_frames == motion_clip.n_frames
    assert np.array_equal(cloned_clip.joint_rotmat, motion_clip.joint_rotmat)
    assert np.array_equal(
        cloned_clip.root_world_position, motion_clip.root_world_position)
    assert cloned_clip.restpose_name == motion_clip.restpose_name
    assert cloned_clip.motion_record_id == motion_clip.motion_record_id
    assert cloned_clip.joint_names == motion_clip.joint_names
    assert cloned_clip.cutoff_frames == motion_clip.cutoff_frames
    assert cloned_clip.cutoff_ranges == motion_clip.cutoff_ranges
    assert cloned_clip.app_name == motion_clip.app_name

def test_slice(motion_clip):
    """Test slicing a MotionClip."""
    sliced_clip = motion_clip.slice(2, 5)
    assert sliced_clip.n_frames == 3
    assert np.array_equal(sliced_clip.joint_rotmat, motion_clip.joint_rotmat[2:5])
    assert sliced_clip.restpose_name == motion_clip.restpose_name
    assert sliced_clip.motion_record_id is None
    assert sliced_clip.joint_names == motion_clip.joint_names
    assert sliced_clip.cutoff_frames[0][0] == 0
    assert sliced_clip.cutoff_frames[1][0] == sliced_clip.n_frames - 1
    assert sliced_clip.cutoff_ranges is not None
    assert len(sliced_clip.cutoff_ranges) > 0
    assert sliced_clip.app_name == motion_clip.app_name

def test_to_dict(motion_clip):
    """Test converting MotionClip to dictionary."""
    clip_dict = motion_clip.to_dict()
    assert clip_dict['joint_names'] == motion_clip.joint_names
    assert clip_dict['len'] == motion_clip.n_frames

def test_to_npz(motion_clip):
    """Test converting MotionClip to npz."""
    npz_file = motion_clip.to_npz()
    assert npz_file is not None

def test_from_dict():
    """Test constructing MotionClip from dictionary."""
    motion_dict = {
        'joint_names': ['joint1', 'joint2'],
        'rotmat': np.random.rand(10, 2, 3, 3),
        'transl': np.random.rand(10, 3),
        'len': 10,
        'restpose_name': 'restpose',
        'app_name': 'python_backend',
        'cutoff_frames': [(0, 1, 2)],
        'cutoff_ranges': [(0, 5, 1, 2)],
    }
    clip = MotionClip.from_dict(motion_dict)
    assert clip.n_frames == 10
    assert clip.restpose_name == 'restpose'
    assert clip.app_name == 'python_backend'

def test_from_npz():
    """Test constructing MotionClip from npz."""
    motion_dict = {
        'joint_names': ['joint1', 'joint2'],
        'rotmat': np.random.rand(10, 2, 3, 3),
        'transl': np.random.rand(10, 3),
        'len': 10,
        'restpose_name': 'restpose',
        'app_name': 'python_backend',
        'cutoff_frames': [(0, 1, 2)],
        'cutoff_ranges': [(0, 5, 1, 2)],
    }
    npz_io = io.BytesIO()
    np.savez_compressed(npz_io, **motion_dict)
    npz_io.seek(0)
    clip = MotionClip.from_npz(npz_io)
    assert clip.n_frames == 10
    assert clip.restpose_name == 'restpose'
    assert clip.app_name == 'python_backend'

def test_dump_and_load_dict(motion_clip):
    """Test dumping and loading MotionClip."""
    clip_dict = motion_clip.to_dict()
    reloaded_clip = MotionClip.from_dict(clip_dict)
    assert reloaded_clip.n_frames == motion_clip.n_frames
    assert np.array_equal(reloaded_clip.joint_rotmat, motion_clip.joint_rotmat)
    assert np.array_equal(
        reloaded_clip.root_world_position, motion_clip.root_world_position)
    assert reloaded_clip.restpose_name == motion_clip.restpose_name
    assert reloaded_clip.motion_record_id == motion_clip.motion_record_id
    assert reloaded_clip.joint_names == motion_clip.joint_names
    assert reloaded_clip.cutoff_frames == motion_clip.cutoff_frames
    assert reloaded_clip.cutoff_ranges == motion_clip.cutoff_ranges
    assert reloaded_clip.app_name == motion_clip.app_name

def test_dump_and_load_npz(motion_clip):
    """Test dumping and loading MotionClip."""
    npz_io = motion_clip.to_npz()
    reloaded_clip = MotionClip.from_npz(npz_io)
    assert reloaded_clip.n_frames == motion_clip.n_frames
    assert np.array_equal(reloaded_clip.joint_rotmat, motion_clip.joint_rotmat)
    assert np.array_equal(
        reloaded_clip.root_world_position, motion_clip.root_world_position)
    assert reloaded_clip.restpose_name == motion_clip.restpose_name
    assert reloaded_clip.motion_record_id == motion_clip.motion_record_id
    assert reloaded_clip.joint_names == motion_clip.joint_names
    assert reloaded_clip.cutoff_frames == motion_clip.cutoff_frames
    assert reloaded_clip.cutoff_ranges == motion_clip.cutoff_ranges
    assert reloaded_clip.app_name == motion_clip.app_name

def test_concat():
    """Test concatenating MotionClips."""
    clip1 = MotionClip(
        n_frames=5,
        joint_names=['joint1', 'joint2'],
        joint_rotmat=np.random.rand(5, 2, 3, 3),
        root_world_position=np.random.rand(5, 3),
        restpose_name='restpose',
        app_name='python_backend'
    )
    clip2 = MotionClip(
        n_frames=5,
        joint_names=['joint1', 'joint2'],
        joint_rotmat=np.random.rand(5, 2, 3, 3),
        root_world_position=np.random.rand(5, 3),
        restpose_name='restpose',
        app_name='python_backend'
    )
    concatenated_clip = MotionClip.concat([clip1, clip2])
    assert concatenated_clip.n_frames == 10
    assert concatenated_clip.restpose_name == 'restpose'
    assert concatenated_clip.app_name == 'python_backend'

def test_concat_with_different_app_name():
    """Test concatenating MotionClips with different app names."""
    clip1 = MotionClip(
        n_frames=5,
        joint_names=['joint1', 'joint2'],
        joint_rotmat=np.random.rand(5, 2, 3, 3),
        root_world_position=np.random.rand(5, 3),
        restpose_name='restpose',
        app_name='python_backend'
    )
    clip2 = MotionClip(
        n_frames=5,
        joint_names=['joint1', 'joint2'],
        joint_rotmat=np.random.rand(5, 2, 3, 3),
        root_world_position=np.random.rand(5, 3),
        restpose_name='restpose',
        app_name='babylon'
    )
    with pytest.raises(ValueError):
        MotionClip.concat([clip1, clip2])

# Blendshape related tests
def test_init_with_blendshapes(motion_clip_with_blendshapes):
    """Test the initialization of MotionClip with blendshapes."""
    assert motion_clip_with_blendshapes.n_frames == 10
    assert motion_clip_with_blendshapes.joint_names == ['joint1', 'joint2']
    assert motion_clip_with_blendshapes.joint_rotmat.shape == (10, 2, 3, 3)
    assert motion_clip_with_blendshapes.root_world_position.shape == (10, 3)
    assert motion_clip_with_blendshapes.blendshape_names == \
        ['blendshape1', 'blendshape2', 'blendshape3']
    assert motion_clip_with_blendshapes.blendshape_values.shape == (10, 3)
    assert motion_clip_with_blendshapes.app_name == 'python_backend'

def test_init_without_blendshapes(motion_clip):
    """Test the initialization of MotionClip without blendshapes."""
    assert motion_clip.blendshape_names is None
    assert motion_clip.blendshape_values is None

def test_set_blendshapes(motion_clip):
    """Test setting blendshapes."""
    blendshape_names = ['blendshape1', 'blendshape2']
    blendshape_values = np.random.rand(10, 2)
    motion_clip.set_blendshape(blendshape_names, blendshape_values)
    assert motion_clip.blendshape_names == blendshape_names
    assert np.array_equal(motion_clip.blendshape_values, blendshape_values)

def test_set_blendshapes_none(motion_clip):
    """Test setting blendshapes to None."""
    motion_clip.set_blendshape(None, None)
    assert motion_clip.blendshape_names is None
    assert motion_clip.blendshape_values is None

def test_set_blendshapes_mismatch():
    """Test setting blendshapes with mismatched names and values."""
    n_frames = 10
    joint_names = ['joint1', 'joint2']
    joint_rotmat = np.random.rand(n_frames, len(joint_names), 3, 3)
    root_world_position = np.random.rand(n_frames, 3)
    clip = MotionClip(
        n_frames=n_frames,
        joint_names=joint_names,
        joint_rotmat=joint_rotmat,
        root_world_position=root_world_position,
        restpose_name='restpose'
    )

    # Test with mismatched lengths
    blendshape_names = ['blendshape1', 'blendshape2']
    blendshape_values = np.random.rand(10, 3)  # 3 blendshapes but only 2 names
    with pytest.raises(ValueError):
        clip.set_blendshape(blendshape_names, blendshape_values)

def test_set_blendshapes_values_none_names_not():
    """Test setting blendshape_values to None but names not None."""
    n_frames = 10
    joint_names = ['joint1', 'joint2']
    joint_rotmat = np.random.rand(n_frames, len(joint_names), 3, 3)
    root_world_position = np.random.rand(n_frames, 3)
    clip = MotionClip(
        n_frames=n_frames,
        joint_names=joint_names,
        joint_rotmat=joint_rotmat,
        root_world_position=root_world_position,
        restpose_name='restpose'
    )

    blendshape_names = ['blendshape1', 'blendshape2']
    with pytest.raises(ValueError):
        clip.set_blendshape(blendshape_names, None)

def test_clone_with_blendshapes(motion_clip_with_blendshapes):
    """Test cloning a MotionClip with blendshapes."""
    cloned_clip = motion_clip_with_blendshapes.clone()
    assert cloned_clip.n_frames == motion_clip_with_blendshapes.n_frames
    assert np.array_equal(cloned_clip.joint_rotmat,
                          motion_clip_with_blendshapes.joint_rotmat)
    assert np.array_equal(
        cloned_clip.root_world_position,
        motion_clip_with_blendshapes.root_world_position)
    assert cloned_clip.restpose_name == motion_clip_with_blendshapes.restpose_name
    assert cloned_clip.motion_record_id == motion_clip_with_blendshapes.motion_record_id
    assert cloned_clip.joint_names == motion_clip_with_blendshapes.joint_names
    assert cloned_clip.cutoff_frames == motion_clip_with_blendshapes.cutoff_frames
    assert cloned_clip.cutoff_ranges == motion_clip_with_blendshapes.cutoff_ranges
    assert cloned_clip.app_name == motion_clip_with_blendshapes.app_name
    assert cloned_clip.blendshape_names == motion_clip_with_blendshapes.blendshape_names
    assert np.array_equal(
        cloned_clip.blendshape_values,
        motion_clip_with_blendshapes.blendshape_values)

def test_slice_with_blendshapes(motion_clip_with_blendshapes):
    """Test slicing a MotionClip with blendshapes."""
    sliced_clip = motion_clip_with_blendshapes.slice(2, 5)
    assert sliced_clip.n_frames == 3
    assert np.array_equal(
        sliced_clip.joint_rotmat,
        motion_clip_with_blendshapes.joint_rotmat[2:5])
    assert sliced_clip.restpose_name == motion_clip_with_blendshapes.restpose_name
    assert sliced_clip.motion_record_id is None
    assert sliced_clip.joint_names == motion_clip_with_blendshapes.joint_names
    assert sliced_clip.cutoff_frames[0][0] == 0
    assert sliced_clip.cutoff_frames[1][0] == sliced_clip.n_frames - 1
    assert sliced_clip.cutoff_ranges is not None
    assert len(sliced_clip.cutoff_ranges) > 0
    assert sliced_clip.app_name == motion_clip_with_blendshapes.app_name
    assert sliced_clip.blendshape_names == motion_clip_with_blendshapes.blendshape_names
    assert np.array_equal(
        sliced_clip.blendshape_values,
        motion_clip_with_blendshapes.blendshape_values[2:5])

def test_to_dict_with_blendshapes(motion_clip_with_blendshapes):
    """Test converting MotionClip with blendshapes to dictionary."""
    clip_dict = motion_clip_with_blendshapes.to_dict()
    assert clip_dict['joint_names'] == motion_clip_with_blendshapes.joint_names
    assert clip_dict['len'] == motion_clip_with_blendshapes.n_frames
    assert 'blendshape_names' in clip_dict
    assert 'blendshape_values' in clip_dict
    assert clip_dict['blendshape_names'] == \
        motion_clip_with_blendshapes.blendshape_names
    assert np.array_equal(
        clip_dict['blendshape_values'],
        motion_clip_with_blendshapes.blendshape_values)

def test_to_dict_without_blendshapes(motion_clip):
    """Test converting MotionClip without blendshapes to dictionary."""
    clip_dict = motion_clip.to_dict()
    assert clip_dict['joint_names'] == motion_clip.joint_names
    assert clip_dict['len'] == motion_clip.n_frames
    assert 'blendshape_names' not in clip_dict
    assert 'blendshape_values' not in clip_dict

def test_from_dict_with_blendshapes():
    """Test constructing MotionClip with blendshapes from dictionary."""
    motion_dict = {
        'joint_names': ['joint1', 'joint2'],
        'rotmat': np.random.rand(10, 2, 3, 3),
        'transl': np.random.rand(10, 3),
        'len': 10,
        'restpose_name': 'restpose',
        'app_name': 'python_backend',
        'cutoff_frames': [(0, 1, 2)],
        'cutoff_ranges': [(0, 5, 1, 2)],
        'blendshape_names': ['blendshape1', 'blendshape2'],
        'blendshape_values': np.random.rand(10, 2),
    }
    clip = MotionClip.from_dict(motion_dict)
    assert clip.n_frames == 10
    assert clip.restpose_name == 'restpose'
    assert clip.app_name == 'python_backend'
    assert clip.blendshape_names == ['blendshape1', 'blendshape2']
    assert clip.blendshape_values.shape == (10, 2)

def test_from_npz_with_blendshapes():
    """Test constructing MotionClip with blendshapes from npz."""
    motion_dict = {
        'joint_names': ['joint1', 'joint2'],
        'rotmat': np.random.rand(10, 2, 3, 3),
        'transl': np.random.rand(10, 3),
        'len': 10,
        'restpose_name': 'restpose',
        'app_name': 'python_backend',
        'cutoff_frames': [(0, 1, 2)],
        'cutoff_ranges': [(0, 5, 1, 2)],
        'blendshape_names': ['blendshape1', 'blendshape2'],
        'blendshape_values': np.random.rand(10, 2),
    }
    npz_io = io.BytesIO()
    np.savez_compressed(npz_io, **motion_dict)
    npz_io.seek(0)
    clip = MotionClip.from_npz(npz_io)
    assert clip.n_frames == 10
    assert clip.restpose_name == 'restpose'
    assert clip.app_name == 'python_backend'
    assert clip.blendshape_names == ['blendshape1', 'blendshape2']
    assert clip.blendshape_values.shape == (10, 2)

def test_dump_and_load_dict_with_blendshapes(motion_clip_with_blendshapes):
    """Test dumping and loading MotionClip with blendshapes."""
    clip_dict = motion_clip_with_blendshapes.to_dict()
    reloaded_clip = MotionClip.from_dict(clip_dict)
    assert reloaded_clip.n_frames == motion_clip_with_blendshapes.n_frames
    assert np.array_equal(reloaded_clip.joint_rotmat,
                          motion_clip_with_blendshapes.joint_rotmat)
    assert np.array_equal(
        reloaded_clip.root_world_position,
        motion_clip_with_blendshapes.root_world_position)
    assert reloaded_clip.restpose_name == motion_clip_with_blendshapes.restpose_name
    assert reloaded_clip.motion_record_id == \
        motion_clip_with_blendshapes.motion_record_id
    assert reloaded_clip.joint_names == motion_clip_with_blendshapes.joint_names
    assert reloaded_clip.cutoff_frames == motion_clip_with_blendshapes.cutoff_frames
    assert reloaded_clip.cutoff_ranges == motion_clip_with_blendshapes.cutoff_ranges
    assert reloaded_clip.app_name == motion_clip_with_blendshapes.app_name
    assert reloaded_clip.blendshape_names == \
        motion_clip_with_blendshapes.blendshape_names
    assert np.array_equal(
        reloaded_clip.blendshape_values,
        motion_clip_with_blendshapes.blendshape_values)

def test_dump_and_load_npz_with_blendshapes(motion_clip_with_blendshapes):
    """Test dumping and loading MotionClip with blendshapes."""
    npz_io = motion_clip_with_blendshapes.to_npz()
    reloaded_clip = MotionClip.from_npz(npz_io)
    assert reloaded_clip.n_frames == motion_clip_with_blendshapes.n_frames
    assert np.array_equal(
        reloaded_clip.joint_rotmat,
        motion_clip_with_blendshapes.joint_rotmat)
    assert np.array_equal(
        reloaded_clip.root_world_position,
        motion_clip_with_blendshapes.root_world_position)
    assert reloaded_clip.restpose_name == motion_clip_with_blendshapes.restpose_name
    assert reloaded_clip.motion_record_id == \
        motion_clip_with_blendshapes.motion_record_id
    assert reloaded_clip.joint_names == motion_clip_with_blendshapes.joint_names
    assert reloaded_clip.cutoff_frames == motion_clip_with_blendshapes.cutoff_frames
    assert reloaded_clip.cutoff_ranges == motion_clip_with_blendshapes.cutoff_ranges
    assert reloaded_clip.app_name == motion_clip_with_blendshapes.app_name
    assert reloaded_clip.blendshape_names == \
    motion_clip_with_blendshapes.blendshape_names
    assert np.array_equal(
        reloaded_clip.blendshape_values,
        motion_clip_with_blendshapes.blendshape_values)

def test_concat_with_blendshapes():
    """Test concatenating MotionClips with blendshapes."""
    clip1 = MotionClip(
        n_frames=5,
        joint_names=['joint1', 'joint2'],
        joint_rotmat=np.random.rand(5, 2, 3, 3),
        root_world_position=np.random.rand(5, 3),
        restpose_name='restpose',
        app_name='python_backend',
        blendshape_names=['blendshape1', 'blendshape2'],
        blendshape_values=np.random.rand(5, 2)
    )
    clip2 = MotionClip(
        n_frames=5,
        joint_names=['joint1', 'joint2'],
        joint_rotmat=np.random.rand(5, 2, 3, 3),
        root_world_position=np.random.rand(5, 3),
        restpose_name='restpose',
        app_name='python_backend',
        blendshape_names=['blendshape1', 'blendshape2'],
        blendshape_values=np.random.rand(5, 2)
    )
    concatenated_clip = MotionClip.concat([clip1, clip2])
    assert concatenated_clip.n_frames == 10
    assert concatenated_clip.restpose_name == 'restpose'
    assert concatenated_clip.app_name == 'python_backend'
    assert concatenated_clip.blendshape_names == ['blendshape1', 'blendshape2']
    assert concatenated_clip.blendshape_values.shape == (10, 2)

def test_concat_with_different_blendshape_names():
    """Test concatenating MotionClips with different blendshape names."""
    clip1 = MotionClip(
        n_frames=5,
        joint_names=['joint1', 'joint2'],
        joint_rotmat=np.random.rand(5, 2, 3, 3),
        root_world_position=np.random.rand(5, 3),
        restpose_name='restpose',
        app_name='python_backend',
        blendshape_names=['blendshape1', 'blendshape2'],
        blendshape_values=np.random.rand(5, 2)
    )
    clip2 = MotionClip(
        n_frames=5,
        joint_names=['joint1', 'joint2'],
        joint_rotmat=np.random.rand(5, 2, 3, 3),
        root_world_position=np.random.rand(5, 3),
        restpose_name='restpose',
        app_name='python_backend',
        blendshape_names=['blendshape1', 'blendshape3'],  # Different blendshape names
        blendshape_values=np.random.rand(5, 2)
    )
    with pytest.raises(ValueError):
        MotionClip.concat([clip1, clip2])

def test_concat_mixed_blendshapes():
    """Test concatenating MotionClips where
    one has blendshapes and the other doesn't."""
    clip1 = MotionClip(
        n_frames=5,
        joint_names=['joint1', 'joint2'],
        joint_rotmat=np.random.rand(5, 2, 3, 3),
        root_world_position=np.random.rand(5, 3),
        restpose_name='restpose',
        app_name='python_backend',
        blendshape_names=['blendshape1', 'blendshape2'],
        blendshape_values=np.random.rand(5, 2)
    )
    clip2 = MotionClip(
        n_frames=5,
        joint_names=['joint1', 'joint2'],
        joint_rotmat=np.random.rand(5, 2, 3, 3),
        root_world_position=np.random.rand(5, 3),
        restpose_name='restpose',
        app_name='python_backend'
        # No blendshapes
    )
    with pytest.raises(ValueError):
        MotionClip.concat([clip1, clip2])

def test_concat_both_without_blendshapes():
    """Test concatenating MotionClips without blendshapes."""
    clip1 = MotionClip(
        n_frames=5,
        joint_names=['joint1', 'joint2'],
        joint_rotmat=np.random.rand(5, 2, 3, 3),
        root_world_position=np.random.rand(5, 3),
        restpose_name='restpose',
        app_name='python_backend'
    )
    clip2 = MotionClip(
        n_frames=5,
        joint_names=['joint1', 'joint2'],
        joint_rotmat=np.random.rand(5, 2, 3, 3),
        root_world_position=np.random.rand(5, 3),
        restpose_name='restpose',
        app_name='python_backend'
    )
    concatenated_clip = MotionClip.concat([clip1, clip2])
    assert concatenated_clip.n_frames == 10
    assert concatenated_clip.restpose_name == 'restpose'
    assert concatenated_clip.app_name == 'python_backend'
    assert concatenated_clip.blendshape_names is None
    assert concatenated_clip.blendshape_values is None
