# BCC_RCM

Basal Cell Carcinoma by Reflectance Confocal Microscopy images.

## 1. Prepare dataset
For current experiment, the dataset is well provided, and well classified in different folders.

### 1.1. Terminations
About the dataset, there are some terms to be explained.

**image sequence**: In hospital when doctors are diagnosing the skin conditions of patient, the doctor may  take
a sequence of images (RCM images) for the lesions on the skin. we call such sequence of images ``image sequence``.

**leaf folder (or leaf directory)**: This is about the folder structure of the dataset. In the dataset, one patient
may have multiple sequences of images, and different image sequence is located in different sub-folder. And 
``leaf folder`` is the folder who contains a single image sequence. 

For example, folder ``"1412 R chest R jaw BCC/"`` has such structure:
```code
harddisk/dataset/1412 R chest R jaw BCC/
    1412/
        R chest/
            Confocal Images/
            VivaStack #1/
            VivaStack #2/
            VivaStack #3/
        R jawline/
            Confocal Images/
            VivaStack #1/
            VivaStack #2/
            VivaStack #3/
```
And ``leaf folders`` are: "Confocal Images", "VivaStack #1", "VivaStack #2", "VivaStack #3".

### 1.2. Assumptions
Of the dataset, we have two important assumptions.

**Assumption 1**: Different folders have different images. And there are no duplicated images between folders.

**Assumption 2**: Each leaf folder exclusively contains images for a single sequence, and the images of each sequence
are exclusively located within a single leaf folder.

## 2. Split dataset for training and validation
Specifically, we assume the dataset being in folder ``./harddisk/dataset/``. And then we split the dataset
into two parts: training dataset and testing dataset (or validation dataset).

While splitting, we identified the images with 5 labels:
* BCC (Basal Cell Carcinoma)
* NS (normal skin)
* Melanocytic
* Lentigo
* Seb K

Since our purpose is to distinguish BCC images, we take all the other 4 labels as ``other``.

All the splitting logic is in [main_classify_dir_by_patient.py](./main_classify_dir_by_patient.py) .
We will explain the steps in the following parts.

### 2.1. Step 1: split folders by patient

We make sure that, training data and testing data belong to different patients. And the patients of testing data
will not involve in the training process. The result of this step will be some files:
```
load_by_patient_bcc_dirs.txt
load_by_patient_other_dirs.txt
load_by_patient_unknown_dirs.txt
load_by_seq_bcc_dirs.txt
load_by_seq_other_dirs.txt
```
Since we take "melanocytic", "lentigo", "ns" and "seb K" as ``other``, hereby the splitting result
has only two types: ``BCC`` and ``other``.
The "``unknown``" file contains folders of unknown disease type, and we can ignore it for now. 
Those "unknown" folders won't involve in the training and testing process.

Please note, the first three folders in the above files are in ``patient`` level. 
One patient may have multiple image sequences, and therefore have multiple leaf folders.
And the last two folders are in ``sequence`` level, containing the leaf folders respectively.

### 2.2. Step 2: split folders by training and testing

We split ``BCC`` folders and ``other`` folders into training and testing parts, separately.
The result of this step is 4 files:
```
dir_train_bcc.txt
dir_train_other.txt
dir_val_bcc.txt
dir_val_other.txt
```
Please note, the folders in the above files are in ``sequence`` level. That means each folder is a leaf folder.

| split       | sequence count<br/>(leaf dir count) | image count |
|-------------|-------------------------------------|-------------|
| train_bcc   | 630                                 | 17187       |
| train_other | 1103                                | 31085       |
| test_bcc    | 137                                 | 3865        |
| test_other  | 207                                 | 5843        |
| **sum**     | 2077                                | 57980       |


To get some statistical data, we also output the leaf folders by disease label. And they are in the following
files:
```
load_stat_bcc_dirs.txt
load_stat_lentigo_dirs.txt
load_stat_melanocytic_dirs.txt
load_stat_ns_dirs.txt
load_stat_sebK_dirs.txt
```

### 2.3. Splitting result
Overall, after splitting, the ``"harddisk/"`` folder structure should be like this:
```
./harddisk/
  dataset/
  dir_train_bcc.txt
  dir_train_other.txt
  dir_val_bcc.txt
  dir_val_other.txt
  load_by_patient_bcc_dirs.txt
  load_by_patient_other_dirs.txt
  load_by_patient_unknown_dirs.txt
  load_by_seq_bcc_dirs.txt
  load_by_seq_other_dirs.txt
  load_stat_bcc_dirs.txt
  load_stat_lentigo_dirs.txt
  load_stat_melanocytic_dirs.txt
  load_stat_ns_dirs.txt
  load_stat_sebK_dirs.txt
```

## 3. How to run
For training, the entry file is [train_harddisk.py](./train_harddisk.py). And here is how to start it.
```bash
gpu_ids="0 1 2 3"
image_size="224 224"
batch_size=128
num_workers=4
dir_code="./code"
data_dir="./dataset/harddisk"
ifile_train_bcc="dir_train_bcc.txt"
ifile_train_other="dir_train_other.txt"
ifile_test_bcc="dir_val_bcc.txt"
ifile_test_other="dir_val_other.txt"

python -u ./train_harddisk.py               \
    --lr 0.0001                             \
    --epoch 40                              \
    --seed 1234                             \
    --batch_size $batch_size                \
    --image_size $image_size                \
    --num_workers $num_workers              \
    --log_interval 20                       \
    --ckpt_save_interval 5                  \
    --ckpt_save_dir "./checkpoint"          \
    --ckpt_load_path ""                     \
    --data_dir $data_dir                    \
    --ifile_train_bcc   $ifile_train_bcc    \
    --ifile_train_other $ifile_train_other  \
    --ifile_test_bcc    $ifile_test_bcc     \
    --ifile_test_other  $ifile_test_other   \
    --gpu_ids $gpu_ids
```

## 4. Notes
This project utilizes the [torchvision.models.resnet](https://pytorch.org/vision/main/models/resnet.html) module
for BCC image classification. To enhance code readability, we have duplicated the 
[ResNet](https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html) 
code from [torchvision](https://pytorch.org/vision/stable/index.html) package and stored it locally within our 
`models` directory.

### 4.1. Notes about input image size
Through experiments, we observed that, the input image size does not impact the processing logic or procedure of
`ResNet`. Instead, it solely affects the processing speed. Below are the procedures for 
[ResNet101](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#torchvision.models.resnet101). 
The input image size is 256x256 pixels, and the output consists of a two-element vector for each image. Given that BCC 
classification produces a binary result (1 or 0), the output vector contains only two elements.
```Python
def _forward_impl(self, x: Tensor) -> Tensor:
    # assume the input x has shape [10, 3, 256, 256], where 10 is batch size
    x = self.conv1(x)       # from [10,  3, 256, 256] to [10, 64, 128, 128]
    x = self.bn1(x)         # from [10, 64, 128, 128] to [10, 64, 128, 128]
    x = self.relu(x)        # from [10, 64, 128, 128] to [10, 64, 128, 128]
    x = self.maxpool(x)     # from [10, 64, 128, 128] to [10, 64,  64,  64]

    x = self.layer1(x)      # from [10, 64,   64, 64] to [10, 256,  64, 64]
    x = self.layer2(x)      # from [10, 256,  64, 64] to [10, 512,  32, 32]
    x = self.layer3(x)      # from [10, 512,  32, 32] to [10, 1024, 16, 16]
    x = self.layer4(x)      # from [10, 1024, 16, 16] to [10, 2048,  8,  8]

    x = self.avgpool(x)     # from [10, 2048, 8, 8]   to [10, 2048, 1, 1]
    x = torch.flatten(x, 1) # from [10, 2048, 1, 1]   to [10, 2048]
    x = self.fc(x)          # from [10, 2048]         to [10, 2]

    return x
```

## 5. Training result accuracy
As shown in the source code, the default learning rate is 0.0001 and lr schedule is ``CosineAnnealingLR``.
During the experiments, we employed the ``resnet101`` and ``resnet152`` models. Their parameter counts are
40.54 million and 55.45 million, respectively.
By differing the input image size and augmentation options, we get the accuracy on testing dataset:

| input size | Resnet    | BCC+Other  | BCC    | Other  | happen<br/>at epoch | augment                                                   |
|------------|-----------|------------|--------|--------|---------------------|-----------------------------------------------------------|
| 224x224    | resnet101 |            |        |        |                     | RandomHorizontalFlip(p=0.5)                               |
| 500x500    | resnet101 |            |        |        |                     | RandomHorizontalFlip(p=0.5)                               |
| 1000x1000  | resnet101 |            |        |        |                     | RandomHorizontalFlip(p=0.5)                               |
| 224x224    | resnet152 |            |        |        |                     | RandomHorizontalFlip(p=0.5)                               |
| 500x500    | resnet152 |            |        |        |                     | RandomHorizontalFlip(p=0.5)                               |
| 1000x1000  | resnet152 |            |        |        |                     | RandomHorizontalFlip(p=0.5)                               |
| 224x224    | resnet152 | **0.9154** | 0.8823 | 0.9352 | 27 of 30            | RandomHorizontalFlip(p=0.5)<br/>RandomRotation((-15, 15)) |
| 500x500    | resnet152 | **0.9485** | 0.9705 | 0.9352 | 19 of 20            | RandomHorizontalFlip(p=0.5)<br/>RandomRotation((-15, 15)) |
| 1000x1000  | resnet152 | **0.9462** | 0.9090 | 0.9644 | 7 of 10             | RandomHorizontalFlip(p=0.5)<br/>RandomRotation((-15, 15)) |
