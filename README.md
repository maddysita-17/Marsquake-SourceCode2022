# Marsquake-SourceCode2022

Necessary packages: Obspy, NumPy, Pandas and Matplotlib 

1. Once Obspy is installed model_build needs to be run either in Jupyter Notebook or Python to create the Obspy formatted interior structure models.
2. seimogram_plotter will pull from "fwdcalc" folder to create Figures 4, 6, 8 and B1
3. Running fault-guess-XXXX.py will produce a csv file in the "event-by-event" folder of all the possible soultions over a range of depth and interior structure model
4. Beach_plotter.ipynb must be run in Jupyter notebook and will pull the csv file from the "even-by-event" folder and produce the beachballs of the complete solution set for each event shown in Figures 5, 7, 9 and B3.
