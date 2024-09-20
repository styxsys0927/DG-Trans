## How to run:

Before running the code, make sure you uodate the data directory in the load_incident() function in load_incident.py.

To run our DG-trans model, ```cd``` in to the project root directory and use the command:
```
python train_incident_gt.py
```

The conda environment below was tested:
```
python = 3.9.7
numpy = 1.20.3
pandas = 21.0
cudatoolkit = 11.3.1
scikit-learn = 0.24.2
pytorch = 1.11.0
torchaudio = 0.11.0
torchvision = 0.12.0 
```
