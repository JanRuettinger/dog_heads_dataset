# Dog Dataset

## Initial purpose of the dataset

## Numbers of the dataset

## How can I download the dataset

### Install libs for preprocessing
`conda create --name <envname> --file env.yml`

### Download tsinghua dataset
You can download the original Tsinghua Dogs Dataset here: https://cg.cs.tsinghua.edu.cn/ThuDogs/

### Run preprocessing step 1
Run notebook `step_1.ipynb` to crop images according to bounding boxes provided by Tsinghua Dogs Dataset creators

### Run preprocessing step 2 (uses provided notebook)
Run notebook `step_2.ipynb` to refine the crops and filter out side views

### Train, validation & test split
Run the notebook `step_3.ipynb` to split the data into train and validation sets