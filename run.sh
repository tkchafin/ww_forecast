

# fitting variants + qPCR to hospital occupancy 
# lineage-level variants resolution
# python3 ./forecast.py \
#     -l variants_data/lineage_all_weighted_smoothed7.csv \
#     -p PNS_data/Daily_COVID-19_hospital_occupancy.formatted.csv \
#     --prevalence_col "SevenDayAverage" \
#     --peak_threshold 0.02 \
#     --threshold 0.01 \
#     --window_width 30  \
#     -ub lineage_metadata/usher_barcodes.csv \
#     --consensus mode \
#     -r annotations/structural_only.bed  \
#     --prefix test_all \
#     -f test_all_features.csv \
#     --extra_features prevalence_data/nationalavg.csv

# fitting variants + qPCR to hospital occupancy with 'level2' collapsing 
python3 ./forecast.py \
    -l variants_data/lineage_lv1_weighted_smoothed7.csv \
    -p PNS_data/Daily_COVID-19_hospital_occupancy.formatted.csv \
    --prevalence_col "SevenDayAverage" \
    --peak_threshold 0.05 \
    --threshold 0.01 \
    --window_width 30  \
    -ub lineage_metadata/usher_barcodes.csv \
    --consensus mode \
    -r annotations/structural_only.bed  \
    --prefix test_lv1 \
    -f test_lv1_features.csv \
    --extra_features prevalence_data/nationalavg.csv

# fitting variants to qPCR data 
# lineage-level variants 
# python3 ./forecast.py \
#     -l variants_data/lineage_all_weighted_smoothed7.csv \
#     -p prevalence_data/nationalavg.csv \
#     --prevalence_col "WWAvgMgc" \
#     --peak_threshold 0.01 \
#     --threshold 0.001 \
#     --window_width 30  \
#     -ub lineage_metadata/usher_barcodes.csv \
#     --consensus mode \
#     -r annotations/structural_only.bed  \
#     -f test_all_features.csv \
#     --prefix test_q_all 