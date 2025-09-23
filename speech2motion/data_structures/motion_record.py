from enum import Enum

from speech2motion.utils.log import setup_logger

from ..utils.super import Super
from .annotations import Loopable, MotionKeyword, Random, SpeechKeyword


class MotionRecordType(Enum):
    """Motion record type enumeration.

    Defines different types of motion records in database
    for motion classification and retrieval.
    """
    IDLE_LONG = "idle_long"
    MOTION_KEYWORD = "motion_keyword"
    SPEECH_KEYWORD = "speech_keyword"
    LOOPABLE = "loopable"
    RANDOM = "random"


class MotionRecord(Super):
    """A motion record in database.

    Essential data required for retrieval process will be stored in this class.
    """

    def __init__(self,
                 motion_record_id: int,
                 n_frames: int,
                 startup_frame: int,
                 recovery_frame: int,
                 avatar_name: str,
                 is_idle_long: bool = False,
                 fps: float = 30.0,
                 labels: list[str] | None = None,
                 logger_cfg: None | dict = None) -> None:
        """Initialize motion record object.

        Args:
            motion_record_id (int):
                Primary key motion_record_id of motion in database.
            n_frames (int):
                Total number of frames in the motion.
            startup_frame (int):
                Frame index where motion startup ends.
            recovery_frame (int):
                Frame index where motion recovery starts.
            avatar_name (str):
                Avatar name of the motion.
            is_idle_long (bool, optional):
                Whether it is a long idle motion. Defaults to False.
            fps (float, optional):
                Frame rate of the motion. Defaults to 30.0.
            labels (list[str], optional):
                List of labels required for constrained retrieval of motion.
                Defaults to None, corresponding to empty list.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use class name. Defaults to None.
        """
        Super.__init__(self, logger_cfg)
        logger_name = f'{self.__class__.__name__}_{motion_record_id}'
        self.logger_cfg['logger_name'] = logger_name
        self.logger = setup_logger(**self.logger_cfg)
        self.motion_record_id = motion_record_id
        self.n_frames = n_frames
        self.startup_frame = startup_frame
        self.recovery_frame = recovery_frame
        self.avatar_name = avatar_name
        self.fps = fps
        self._is_idle_long = is_idle_long
        self.loopable: Loopable | None = None
        self.random: Random | None = None
        self.motion_keyword: MotionKeyword | None = None
        self.speech_keyword: SpeechKeyword | None = None
        self.labels: list[str] = labels if labels is not None else []
        self.startup_frame = startup_frame
        self.recovery_frame = recovery_frame

    @property
    def is_idle_long(self) -> bool:
        """Check if motion is a long idle motion.

        Returns:
            bool: True if motion is a long idle motion, False otherwise.
        """
        return self._is_idle_long

    def set_loopable(self, loopable: Loopable) -> None:
        """Set loop information for motion.

        Args:
            loopable (Loopable): Loopable object containing loop information.
        """
        self.loopable = loopable

    def is_loopable(self) -> bool:
        """Check if motion is a loopable motion.

        Returns:
            bool: True if motion is a loopable motion, False otherwise.
        """
        return self.loopable is not None

    def set_random(self, random: Random) -> None:
        """Set random information for motion.

        Args:
            random (Random): Random object containing random information.
        """
        self.random = random

    def is_random(self) -> bool:
        """Check if motion is a random motion.

        Returns:
            bool: True if motion is a random motion, False otherwise.
        """
        return self.random is not None

    def set_motion_keyword(self, motion_keyword: MotionKeyword) -> None:
        """Set keyword information for motion.

        Args:
            motion_keyword (MotionKeyword):
                MotionKeyword object containing keyword information.
        """
        self.motion_keyword = motion_keyword

    def is_motion_keyword(self) -> bool:
        """Check if motion has keyword information.

        Returns:
            bool: True if motion is a keyword motion, False otherwise.
        """
        return self.motion_keyword is not None

    def set_speech_keyword(self, speech_keyword: SpeechKeyword) -> None:
        """Set speech keyword information for motion.

        Args:
            speech_keyword (SpeechKeyword):
                SpeechKeyword object containing speech keyword information.
        """
        self.speech_keyword = speech_keyword

    def is_speech_keyword(self) -> bool:
        """Check if motion has speech keyword information.

        Returns:
            bool: True if motion is a speech keyword motion, False otherwise.
        """
        return self.speech_keyword is not None

    def get_labels(self) -> list[str]:
        """Get label list of motion.

        Returns:
            list[str]: Label list of motion.
        """
        return self.labels

    def to_dict(self) -> dict:
        """Convert MotionRecord object to dictionary.

        Returns:
            dict: Attribute dictionary of MotionRecord object.
        """
        ret_dict = dict(
            motion_record_id=self.motion_record_id,
            n_frames=self.n_frames,
            startup_frame=self.startup_frame,
            recovery_frame=self.recovery_frame,
            avatar_name=self.avatar_name,
            is_idle_long=self._is_idle_long,
            fps=self.fps,
            labels=self.labels,
        )
        if self.is_loopable():
            ret_dict["loopable"] = self.loopable.to_dict()
        if self.is_random():
            ret_dict["random"] = self.random.to_dict()
        if self.is_motion_keyword():
            ret_dict["motion_keyword"] = self.motion_keyword.to_dict()
        if self.is_speech_keyword():
            ret_dict["speech_keyword"] = self.speech_keyword.to_dict()
        return ret_dict

    @classmethod
    def from_dict(cls, data: dict, logger_cfg: None | dict = None) -> 'MotionRecord':
        """Create MotionRecord object from dictionary.

        Args:
            data (dict): Dictionary containing MotionRecord object attributes.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Defaults to None.

        Returns:
            MotionRecord: MotionRecord object created from dictionary.
        """
        motion_record_id = data["motion_record_id"]
        n_frames = data["n_frames"]
        startup_frame = data["startup_frame"]
        recovery_frame = data["recovery_frame"]
        avatar_name = data["avatar_name"]
        is_idle_long = data["is_idle_long"]
        fps = data["fps"]
        labels = data.get("labels", list())
        startup_frame = data.get("startup_frame", None)
        recovery_frame = data.get("recovery_frame", None)
        motion_record = cls(
            motion_record_id=motion_record_id,
            n_frames=n_frames,
            startup_frame=startup_frame,
            recovery_frame=recovery_frame,
            avatar_name=avatar_name,
            is_idle_long=is_idle_long,
            fps=fps,
            labels=labels,
            logger_cfg=logger_cfg
        )
        if "loopable" in data:
            motion_record.set_loopable(Loopable.from_dict(data["loopable"], logger_cfg))
        if "random" in data:
            motion_record.set_random(Random.from_dict(data["random"], logger_cfg))
        if "motion_keyword" in data:
            motion_record.set_motion_keyword(
                MotionKeyword.from_dict(data["motion_keyword"], logger_cfg))
        if "speech_keyword" in data:
            motion_record.set_speech_keyword(
                SpeechKeyword.from_dict(data["speech_keyword"], logger_cfg))
        return motion_record

    def clone(self) -> 'MotionRecord':
        """Clone MotionRecord object.

        Returns:
            MotionRecord: Deep copy of current object.
        """
        motion_record = MotionRecord(
            motion_record_id=self.motion_record_id,
            n_frames=self.n_frames,
            startup_frame=self.startup_frame,
            recovery_frame=self.recovery_frame,
            avatar_name=self.avatar_name,
            is_idle_long=self._is_idle_long,
            fps=self.fps,
            labels=self.labels,
            logger_cfg=self.logger_cfg,
        )
        if self.is_loopable():
            motion_record.set_loopable(self.loopable.clone())
        if self.is_random():
            motion_record.set_random(self.random.clone())
        if self.is_motion_keyword():
            motion_record.set_motion_keyword(self.motion_keyword.clone())
        if self.is_speech_keyword():
            motion_record.set_speech_keyword(self.speech_keyword.clone())
        return motion_record
