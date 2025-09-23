SELECT
    *
FROM
    motion_record mr 
    LEFT JOIN motion_file mf ON mr.motion_file_id = mf.motion_file_id
    LEFT JOIN origin ori ON mf.origin_id = ori.origin_id
    LEFT JOIN armature ON mf.armature_id = armature.armature_id
    LEFT JOIN avatar ON armature.avatar_id = avatar.avatar_id
    LEFT JOIN motion_keyword mk ON mk.motion_keyword_id = mr.motion_keyword_id
    LEFT JOIN speech_keyword sk ON sk.speech_keyword_id = mr.speech_keyword_id
    LEFT JOIN random ra ON ra.random_id = mr.random_id
    LEFT JOIN loopable la ON la.loop_id = mr.loop_id;