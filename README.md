# Code paper Baseline ViSL1
# Requirements
- Python >= 3.6
- GPU 12GB
- Disk >= 100 GB
# Training 
1. Install libary
   Clone this repo
   ```bash
   git clone https://github.com/phu2007vis/Baseline_ViSL1.git
   ```
   Before install libary , assert python is available in your machine
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```
3. Data preparetion
   Before run scrip ,prepare data like this structure:
   ```
   root_folder
         -A1P1
            --rgb
               ---file1.avi
               ---file2.avi
               ---fileN.avi
         -A1P2
            rgb:
               ---file1.avi
               ---file2.avi
               ---fileN.avi
         -A1PN:
            ...
         -A2P1:
            ...
         -A2PN
            ...
         ...
         -ANPN
            ...
   Change the path of data root in resources/utils/data_preparation.py
   ```
   

4. Tranning
