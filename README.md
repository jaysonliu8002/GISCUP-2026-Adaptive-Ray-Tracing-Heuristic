# GIS Cup 2026 solver — configuration reference

**Last updated 8/7/2026**

Sample testing instructions:

**PART 1**

1.In settings, set `ADAPTIVE_DOUBLINGS` = 2 and `GEOJSON_PATH = os.path.join(REPO_ROOT, "dataset1_sample.geojson")`

2.Look for a table like the following below and copy paste into a document

    tau     k    served   swaps   time
    ----------------------------------------
    0.25     50      1321       0  154.6s
    0.25    500      9157      27  885.3s
    0.25   1000     12368      30 1089.5s
    0.5      50       537       4  259.9s
    0.5     500      4884      30 1910.9s
    0.5    1000      8648      30 2286.3s
    0.75     50       209      10  433.6s
    0.75    500      2246      30 2887.6s
    0.75   1000      4351      30 8383.9s
    ----------------------------------------
    total served: 43721

3.Record the runtime it took and paste below the table, should be at the end like below

`[__main__ total] took xxxxx.xxs`

4.Repeat above steps with `ADAPTIVE_DOUBLINGS` = 5 and

`ADAPTIVE_DOUBLINGS` = 8

5.Repeat above steps (including 4) with `"dataset2_sample.geojson"` and `"dataset3_sample.geojson"`

**PART 2**

TBD

**IMPORTANT**

All switches live in `main/settings.py`

Use `run_giscup.py` to run the program.
