#!/bin/sh

python coregistration_profiling.py --coreg_type super_glue --n_event 1 --test_iteration warm_up_1
python coregistration_profiling.py --coreg_type super_glue --n_event 1 --test_iteration warm_up_2
python coregistration_profiling.py --coreg_type super_glue --n_event 1 --test_iteration 1
python coregistration_profiling.py --coreg_type super_glue --n_event 1 --test_iteration 2
python coregistration_profiling.py --coreg_type super_glue --n_event 1 --test_iteration 3
python coregistration_profiling.py --coreg_type super_glue --n_event 2 --test_iteration 1
python coregistration_profiling.py --coreg_type super_glue --n_event 2 --test_iteration 2
python coregistration_profiling.py --coreg_type super_glue --n_event 2 --test_iteration 3
python coregistration_profiling.py --coreg_type super_glue --n_event 4 --test_iteration 1
python coregistration_profiling.py --coreg_type super_glue --n_event 4 --test_iteration 2
python coregistration_profiling.py --coreg_type super_glue --n_event 4 --test_iteration 3
python coregistration_profiling.py --coreg_type super_glue --n_event 8 --test_iteration 1
python coregistration_profiling.py --coreg_type super_glue --n_event 8 --test_iteration 2
python coregistration_profiling.py --coreg_type super_glue --n_event 8 --test_iteration 3
python coregistration_profiling.py --coreg_type super_glue --n_event 16 --test_iteration 1
python coregistration_profiling.py --coreg_type super_glue --n_event 16  --test_iteration 2
python coregistration_profiling.py --coreg_type super_glue --n_event 16  --test_iteration 3
python coregistration_profiling.py --coreg_type super_glue --n_event 1  --device gpu --test_iteration warm_up_1
python coregistration_profiling.py --coreg_type super_glue --n_event 1  --device gpu --test_iteration warm_up_2
python coregistration_profiling.py --coreg_type super_glue --n_event 1  --device gpu --test_iteration 1
python coregistration_profiling.py --coreg_type super_glue --n_event 1  --device gpu --test_iteration 2
python coregistration_profiling.py --coreg_type super_glue --n_event 1  --device gpu --test_iteration 3
python coregistration_profiling.py --coreg_type super_glue --n_event 2  --device gpu --test_iteration 1
python coregistration_profiling.py --coreg_type super_glue --n_event 2  --device gpu --test_iteration 2
python coregistration_profiling.py --coreg_type super_glue --n_event 2  --device gpu --test_iteration 3
python coregistration_profiling.py --coreg_type super_glue --n_event 4  --device gpu --test_iteration 1
python coregistration_profiling.py --coreg_type super_glue --n_event 4  --device gpu --test_iteration 2
python coregistration_profiling.py --coreg_type super_glue --n_event 4  --device gpu --test_iteration 3
python coregistration_profiling.py --coreg_type super_glue --n_event 8  --device gpu --test_iteration 1
python coregistration_profiling.py --coreg_type super_glue --n_event 8  --device gpu --test_iteration 2
python coregistration_profiling.py --coreg_type super_glue --n_event 8  --device gpu --test_iteration 3
python coregistration_profiling.py --coreg_type super_glue --n_event 16 --device gpu  --test_iteration 1
python coregistration_profiling.py --coreg_type super_glue --n_event 16 --device gpu   --test_iteration 2
python coregistration_profiling.py --coreg_type super_glue --n_event 16 --device gpu   --test_iteration 3


