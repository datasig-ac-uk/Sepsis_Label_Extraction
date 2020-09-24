
static_vars = ['admission_type', 'admittime',
               'dischtime', 'hospital_expire_flag', 'deathtime', 'intime', 'outtime',
               'age', 'gender', 'ethnicity', 'insurance', 'initial_diagnosis',
               'first_careunit', 'last_careunit', 'dbsource', 'los_hospital',
               'hospstay_seq', 'los_icu', 'icustay_seq', 't_suspicion', 't_sofa', 't_sepsis_min',
               't_sepsis_max']

categorical_vars = ['heart_rhythm', 'temp_site', 'verbal_response', 'o2_device', 'vent_mode', 'specimen',
                    'ventilator', 'curr_careunit']

numerical_vars = ['heart_rate',
                  'nbp_sys', 'nbp_dias', 'nbp_mean', 'abp_sys', 'abp_dias', 'abp_mean',
                  'temp_celcius', 'o2sat', 'resp_rate',
                  'resp_rate_spont', 'resp_rate_set', 'resp_rate_total', 'minute_vol',
                  'mean_airway_pressure', 'peak_insp_rate', 'plateau_pressure',
                  'o2flow_chart', 'tidal_vol_set', 'tidal_vol_obs', 'tidal_vol_spon',
                  'peep_set', 'baseexcess', 'bicarbonate_bg', 'totalco2',
                  'carboxyhemoglobin', 'chloride_bg', 'calcium_bg', 'glucose_bg',
                  'hematocrit_bg', 'hemoglobin_bg', 'lactate',
                  'methemoglobin', 'o2flow_lab', 'fio2', 'so2', 'pco2', 'peep', 'ph',
                  'po2', 'potassium_bg', 'requiredo2', 'sodium_bg', 'tidalvolume',
                  'alkalinephos', 'ast', 'bilirubin_direct',
                  'bilirubin_total', 'bun', 'creatinine', 'fibrinogen', 'magnesium',
                  'phosphate', 'platelets', 'ptt', 'tropinin_t', 'tropinin_i', 'wbc',
                  'bicarbonate', 'chloride', 'calcium', 'glucose', 'hematocrit',
                  'hemoglobin', 'potassium', 'sodium']

flags = ['intubated', 'on_vent']
identifier = ['subject_id', 'hadm_id', 'icustay_id', 'floored_charttime']

feature_dict = {'static': static_vars,
                'categorical': categorical_vars,
                'numerical': numerical_vars,
                'flags': flags,
                'identifier': identifier
                }

feature_dict_james = {
    'vitals': ['heart_rate', 'o2sat', 'temp_celcius', 'nbp_sys', 'mean_airway_pressure', 'abp_dias', 'resp_rate'],
    'laboratory': ['baseexcess', 'bicarbonate', 'ast', 'fio2', 'ph', 'pco2', 'so2', 'bun', 'alkalinephos', 'calcium', \
                   'chloride', 'creatinine', 'bilirubin_direct', 'glucose', 'lactate', 'magnesium', 'phosphate', \
                   'potassium', 'bilirubin_total', 'tropinin_i', 'hematocrit', 'hemoglobin', 'wbc', 'ptt', \
                   'fibrinogen', 'platelets'],
    'demographics': ['age', 'gender', 'admittime', 'rolling_los_icu'],
    'derived': ['bun_creatinine', 'partial_sofa', 'shock_index']
}