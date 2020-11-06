# arus-muss command line application

## Usage

```bash
# update the built-in muss model
arus-muss update -p=<places> -t=<target> [--pids=<pids>] [-c=<cores>] [-d]

# predict on new data files using the built-in muss model
arus-muss predict (-m=<model> | -t=<target>) -p=<places> -f=<format> -o=<output> [-s=<srs>] [--pids=<pids>] [-d] FILE_PATH...
```

## Arguments

* FILE_PATH     Test file paths stored in signaligner sensor file format, the order should

## Options

    -s <srs>, --srs <srs>                   Set the sampling rates.
    -p <places>, --placements <places>      Set placements. E.g., "DW,DA".
    -t <target>, --target <target>          Set target task name. E.g."INTENSITY".
    -m <model>, --model <model>             Set model path.
    -f <format>, --format <format>          Set input file format. E.g. "MHEALTH_FORMAT" or "SIGNALIGNER".
    -c <cores>, --cores <cores>             Set the number of cores used. E.g., 4.
    -o <output>, --output <output>          Set output folder.
    --pids <pids>                           Set the participant IDs. E.g., "ALL" or "SPADES_1,SPADES_2".
    -d, --debug                             Turn on debug messages.