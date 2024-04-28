PAD
=====
This is the code of the paper "PAD: Patch-Agnostic Defense against Adversarial Patch Attacks", without parallel acceleration.

Setup
-----
conda create --name pad python=3.8<br>
conda activate pad<br>
pip install -r requirements.txt<br>

Usage
----
python run-PAD.py<br>
Remember to replace the file path with your own.<br>
input_path: the dir path of the attacked images<br>
save_path: the dir path where you want to save the defended images
