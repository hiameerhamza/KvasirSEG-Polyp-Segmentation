# 🧠 Polyp Segmentation using U-Net with Attention & SE Blocks

This project implements a deep learning pipeline for **polyp segmentation** using a custom U-Net architecture enhanced with **Attention Mechanisms** and **Squeeze-and-Excitation (SE) blocks**. 
It is trained on the Kvasir-SEG and CVC-ClinicDB datasets.

---

## 📁 Dataset

Download the [Kvasir-SEG Dataset](https://datasets.simula.no/kvasir-seg/) and [CVC-ClinicBD dataset](https://universe.roboflow.com/teste-mhypc/cvc-clinicdb/)

**Abstract**
Colorectal cancer is considered one of the deadliest diseases, contributing to an alarming increase in annual deaths worldwide, with colorectal polyps recognized as precursors to this malignancy. Early and accurate detection of these polyps is crucial for reducing the mortality rate of colorectal cancer. However, the manual detection of polyps is a time-consuming process and requires the expertise of trained medical professionals. Moreover, it often misses polyps due to their varied size, color, and texture. Computer-aided diagnosis systems offer potential improvements, but they often struggle with precision in complex visual environments. This study presents an enhanced deep learning approach using encoder-decoder architecture for colorectal polyp segmentation to capture and utilize complex feature representations. Our approach introduces an enhanced dual attention mechanism, combining spatial and channel-wise attention to focus precisely on critical features. Channel-wise attention, implemented via an optimized Squeeze-and-Excitation (S&E) block, allows the network to capture comprehensive contextual information and interrelationships among different channels, ensuring a more refined feature selection process. The experimental results showed that the proposed model achieved a mean Intersection over Union (IoU) of 0.9054 and 0.9277, a dice coefficient of 0.9006 and 0.9128, a precision of 0.8985 and 0.9517, a recall of 0.9190 and 0.9094, and an accuracy of 0.9806 and 0.9907 on the Kvasir-SEG and CVC-ClinicDB datasets, respectively. Moreover, the proposed model outperforms the existing state-of-the-art resulting in improved patient outcomes with the potential to enhance the early detection of colorectal polyps.


**Complete Procdedure**
Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/hiameerhamza/kvasir-segmentation.git
cd KvasirSEG-Polyp-Segmentation
```

**Set Up a Virtual Environment**

<pre>python -m venv venv
source venv/bin/activate   # For Windows, use `venv\Scripts\activate`
</pre>
**Install the required dependencies using pip:**
<pre>pip install -r requirements.txt</pre>

**Dataset Setup**
Download the Kvasir-SEG Dataset and arrange the files in the following directory structure:
<pre>KvasirSEG/
├── images/   # Input images
└── masks/    # Ground truth masks</pre>

**Model Training**
Once the virtual environment is set up, and the dataset is properly arranged, you can start training the model by running the main script:
<pre>python Model.py</pre>

The model will start training, and during the process, it will save checkpoints, logs, and visualizations of various training metrics such as loss, accuracy, Dice Coefficient, Precision, and Recall.

**Model Metrics**
- Dice Coefficient
- Mean IoU (Intersection over Union)
- Accuracy
- Precision
- Recall
- Loss

