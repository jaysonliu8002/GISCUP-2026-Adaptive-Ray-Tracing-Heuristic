# GIS Cup 2026 solver — configuration reference

**Last updated 8/7/2026**

All switches live in `main/settings.py`

Use `run_giscup.py` to run the program.

**Competition run instructions**

0a. Make sure `TAUS` and `KS` are correctly hardcoded. Go to https://alowe.github.io/gis-cup-2026-evaluator/ and find `competition-parameters.txt` where it will contian the values

0b. Download the dataset `GIS-cup-competition-dataset.geojson` on the same github page and change `GEOJSON_PATH` to match.

1. Go to line 864. Comment out the try/except block and make sure the only function that runs here is the plotDataset() function. Run the program.
2. Read dataset.png in the file select and look at the numbers on top -> a buildings, (b m x c m extent). Identify the b and c and calculate the long diagonal (b^2+c^2)^0.5.
3. Comment out the plotDataset() and uncomment out the try/except block.
4. Identify the smallest integer n such that 80 \* 2^n >= max_length and set ADAPTIVE_DOUBLINGS to n and VANTAGE_RADIUS to a value no greater than 5120 (Use 2560 if OOM error)
5. For each vantage radius setting, do the following before switching to a new radius
6. Adjust VANTAGE_SEED_PERCENT to values between 0.01 and 0.10 inclusive, then run the program.
7. Save each submission and record in a google sheet to find the best output.
