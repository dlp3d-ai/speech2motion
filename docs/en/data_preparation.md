# Data Preparation

To use Speech2Motion, you need to download the offline motion database and set up the required directory structure.

## Download Motion Database

1. **Download the motion database:**

   - **Google Drive Download:** [motion_data.zip](https://drive.google.com/file/d/112pnjuIuNqADS-fAT6RUIAVPtb3VlWlq/view?usp=drive_link)
   - **Baidu Cloud：** [motion_data.zip](https://pan.baidu.com/s/1YCisRewRQQdYT-GzCZxu-w?pwd=wwqm)
   - According to your network environment, choose the appropriate download method to download the compressed motion database file

2. **Extract and organize the data:**
   - Extract the downloaded file to your project root directory
   - Ensure the following directory structure is created:

```
├─configs
├─data
│  ├─motion_files
│  │  └─motion
│  ├─restpose_npz
│  └─motion_database.db
├─docs
├─speech2motion
└─tools
```

## Directory Structure Explanation

- `data/motion_files`: A folder for storing binary motion files.
- `data/restpose_npz/`: A folder for storing restpose data in NPZ format.
- `data/motion_database.db`: A SQLite file that contains the motion database.
- The `data` directory will be mounted to the Docker container at `/workspace/speech2motion/data`


