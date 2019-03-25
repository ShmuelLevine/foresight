#!/bin/zsh

## Job Information
JOB_ID=""
JOB_DESCRIPTION=""

## Model Settings
MODEL_STRUCTURE="100 200 450 100"
FORECAST_STEPS="1 3 5"

## Program Inputs
INPUT_MATRIX=$(pwd)/Matrix.out
INPUT_TIMESERIES=$(pwd)/Jul2004-Jul2008.csv

## Program Settings
MAX_ITERATIONS=500
POPULATION_SIZE=50000

## DE Settings
CR=0.9
F=0.8
F_DITHER=FALSE
F_MIN=0.5
F_MAX=1.5
INITIAL_RANGE=5.0



# Specify whether to try and model the actual output or just the
# difference between time steps (only 1 can be true)
FIT_USE_DIFFERENCES=TRUE
FIT_USE_ABSOLUTE=FALSE

# Migration Options
MIGRATION_SIZE=200
MIGRATION_TOPOLOGY_LOOP=TRUE
MIGRATION_TOPOLOGY_TORUS=FALSE
MIGRATION_TRIGGER_ITERATIONS=100
MIGRATION_TRIGGER_MEAN_DELTA=0.005

## Output Settings
WRITE_AGG_DATA_TO_COUT=true
AGG_DATA_PATH=/path/to/aggregate_output_file
NODE_DATA_PATH=/path/to/aggregate_output_file
RESULTS_DIRECTORY=/path/to/fitter_results


#####  Run Options - Do not edit below this line #####


--fx:DE:pop_size ${POPULATION_SIZE} \
--fx:DE:CR ${CR} \
--fx:DE:F ${F} \
--fx:DE:F_Dither ${F_DITHER} \
--fx:DE:F_min ${F_MIN} \
--fx:DE:F_max ${F_MAX} \
--fx:DE:initial_range ${INITIAL_RANGE} \
--fx:fitter:iterations ${MAX_ITERATIONS} \
--fx:fitter:input_matrix ${INPUT_MATRIX} \
--fx:fitter:input_timeseries ${INPUT_TIMESERIES} \
--fx:fitter:fit_difference ${FIT_USE_DIFFERENCES} \
--fx:fitter:fit_absolute {$FIT_USE_ABSOLUTE} \
--fx:fitter:difference:absolute ${FIT_DIFFERENCE_ABSOLUTE} \
--fx:fitter:difference:percentage ${FIT_DIFFERENCE_PERCENTAGE} \
--fx:fitter:difference:log ${FIT_DIFFERENCE_LOG} \
--fx:fitter:migration_size ${MIGRATION_SIZE} \
--fx:fitter:migration_topology:loop ${MIGRATION_TOPOLOGY_LOOP} \
--fx:fitter:migration_topology:torus ${MIGRATION_TOPOLOGY_TORUS} \
--fx:fitter:migration_trigger:iterations ${MIGRATION_TRIGGER_ITERATIONS} \
--fx:fitter:migration_trigger:delta_mean ${MIGRATION_TRIGGER_MEAN_DELTA=} \
--fx:fitter:ui:write_agg_data_to_cout ${WRITE_AGG_DATA_TO_COUT} \
--fx:fitter:ui:agg_data_path ${AGG_DATA_PATH} \
--fx:fitter:ui:node_data_path ${NODE_DATA_PATH} \
--fx:model:structure ${MODEL_STRUCTURE} \
--fx:model:forecast_steps ${FORECAST_STEPS} \
--fx::fitter:output_path ${RESULTS_DIRECTORY}
