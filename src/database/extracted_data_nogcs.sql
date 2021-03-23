/*
This script is almost identical to extracted_data.sql. We are joining together the other views that we have created (e.g. for the labs data and the vitals data)
It combines this with some demographic information about the patient
We also take ventilation information about a patient to label the times when they get ventilation (we have a vent flag for when they are on ventilation)

The difference here is that we use the version of the sofa view that does not use the cns component in the total score (pivoted_sofa_nogcs). We still include the row of CNS score for interest, though this can be removed if none of the methods are using the derived sofa scores.

*/

/*
This query creates the final pivoted table. It does so by joining together the other views that we have created (e.g. for the labs data and the vitals data)
It combines this with some demographic information about the patient
We also take ventilation information about a patient to label the times when they get ventilation (we have a vent flag for when they are on ventilation)

*/

  DROP TABLE IF EXISTS extracted_data_nogcs CASCADE;
CREATE TABLE extracted_data_nogcs AS

       -- collect the demographic information of each patient
  WITH patient_info AS
       (
        SELECT ad.subject_id, ad.hadm_id, icustay_id, admission_type, admittime, dischtime
               , hospital_expire_flag, deathtime, intime, outtime
               , ROUND((CAST(ad.admittime AS DATE) - CAST(pat.dob AS DATE)) / 365.242, 4) AS age, gender
               , CASE WHEN ad.ethnicity IN 
               (
                     'WHITE' --  40996
                   , 'WHITE - RUSSIAN' --    164
                   , 'WHITE - OTHER EUROPEAN' --     81
                   , 'WHITE - BRAZILIAN' --     59
                   , 'WHITE - EASTERN EUROPEAN' --     25
               ) THEN 'white'
                 WHEN ad.ethnicity IN
               (
                     'BLACK/AFRICAN AMERICAN' --   5440
                   , 'BLACK/CAPE VERDEAN' --    200
                   , 'BLACK/HAITIAN' --    101
                   , 'BLACK/AFRICAN' --     44
                   , 'CARIBBEAN ISLAND' --      9
                ) THEN 'black'
                 WHEN ad.ethnicity IN
                (
                     'HISPANIC OR LATINO' --   1696
                   , 'HISPANIC/LATINO - PUERTO RICAN' --    232
                   , 'HISPANIC/LATINO - DOMINICAN' --     78
                   , 'HISPANIC/LATINO - GUATEMALAN' --     40
                   , 'HISPANIC/LATINO - CUBAN' --     24
                   , 'HISPANIC/LATINO - SALVADORAN' --     19
                   , 'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)' --     13
                   , 'HISPANIC/LATINO - MEXICAN' --     13
                   , 'HISPANIC/LATINO - COLOMBIAN' --      9
                   , 'HISPANIC/LATINO - HONDURAN' --      4
                ) THEN 'hispanic_latino'
                 WHEN ad.ethnicity IN
                (
                      'ASIAN' --   1509
                    , 'ASIAN - CHINESE' --    277
                    , 'ASIAN - ASIAN INDIAN' --     85
                    , 'ASIAN - VIETNAMESE' --     53
                    , 'ASIAN - FILIPINO' --     25
                    , 'ASIAN - CAMBODIAN' --     17
                    , 'ASIAN - OTHER' --     17
                    , 'ASIAN - KOREAN' --     13
                    , 'ASIAN - JAPANESE' --      7
                    , 'ASIAN - THAI' --      4
                ) THEN 'asian'
                WHEN ad.ethnicity IN
                (
                      'UNKNOWN/NOT SPECIFIED' --   4523
                    , 'UNABLE TO OBTAIN' --    814
                    , 'PATIENT DECLINED TO ANSWER' --    559
                ) THEN 'unknown'
                ELSE 'other' END AS ethnicity
               , insurance, diagnosis AS initial_diagnosis, first_careunit, last_careunit, dbsource 
               , ROUND( (CAST(ad.dischtime AS DATE) - CAST(ad.admittime AS DATE)) , 4) AS los_hospital
               , DENSE_RANK() OVER (PARTITION BY ad.subject_id ORDER BY ad.admittime) AS hospstay_seq
               , ROUND( (CAST(ic.outtime AS DATE) - CAST(ic.intime AS DATE)) , 4) AS los_icu
               , DENSE_RANK() OVER (PARTITION BY ic.hadm_id ORDER BY ic.intime) AS icustay_seq

          FROM admissions ad 
         INNER JOIN patients pat
            ON ad.subject_id = pat.subject_id
         INNER JOIN icustays ic 
            ON ad.hadm_id = ic.hadm_id 
         ORDER BY ad.hadm_id
       )

        -- get a list of all observed times
       , times AS
        (
         SELECT icustay_id, charttime FROM extracted_chart vitals
          UNION
        (SELECT icustay_id, charttime FROM extracted_labs labs)
        )


        -- join the lab and vital data together
       , merged AS
        (
          SELECT times.icustay_id, times.charttime
                , heart_rate, heart_rhythm
                , nbp_sys, nbp_dias, nbp_mean, abp_sys, abp_dias, abp_mean
                , temp_site, temp_celcius
                , verbal_response
                , o2sat, resp_rate, resp_rate_spont, resp_rate_set, resp_rate_total
                , minute_vol, mean_airway_pressure, peak_insp_rate, plateau_pressure
                , vitals.o2flow AS o2flow_chart, o2_device, vent_mode
                , tidal_vol_set, tidal_vol_obs, tidal_vol_spon
                , peep_set, fio2_chart, glucose_chart
                , specimen, baseexcess, bicarbonate_bg, totalco2, carboxyhemoglobin, chloride_bg
                , calcium_bg, glucose_bg, hematocrit_bg, hemoglobin_bg, intubated, lactate
                , methemoglobin, labs.o2flow AS o2flow_lab, fio2, so2, pco2, peep, ph, po2
                , potassium_bg, requiredo2, sodium_bg, temperature AS temperature_bg, tidalvolume
                , ventilator, alkalinephos, ast, bilirubin_direct, bilirubin_total, bun
                , creatinine, fibrinogen, magnesium, phosphate, platelets, ptt, tropinin_t
                , tropinin_i, wbc, bicarbonate, chloride, calcium, glucose, hematocrit
                , hemoglobin, potassium, sodium
      
            FROM times
            LEFT JOIN extracted_chart vitals
              ON times.icustay_id = vitals.icustay_id AND times.charttime = vitals.charttime
            LEFT JOIN extracted_labs labs
              ON times.icustay_id=labs.icustay_id AND times.charttime = labs.charttime  
        )
        
        -- Next we clean the sofa score data
        -- We reduce the number of duplicates in sofa score calculation by grouping them. Note that there appears to still be some discrepencies (in epinephrine_dose, norepinephrine_dose, dopamine_dose, dobutamine_dose). If we group by sofa_24hours scores, we will find that there are 28 times where there is discrepency in the SOFA scores. Most of these are a discrepency of 1, but some have a difference of 4, which is highly relevant as we are looking at an increase of 2 for sepsis
        -- A lot of these discrepencies appear on the boundary where something changed, we will resolve this by taking the maximum
       , sofa_scores AS
        (
         SELECT icustay_id, starttime, endtime
                , MAX(respiration_24hours) AS respiration_24hours
                , MAX(coagulation_24hours) AS coagulation_24hours
                , MAX(liver_24hours) AS liver_24hours
                , MAX(cardiovascular_24hours) AS cardiovascular_24hours
                , MAX(cns_24hours) AS cns_24hours
                , MAX(renal_24hours) AS renal_24hours
                , MAX(sofa_24hours) AS sofa_24hours
           FROM pivoted_sofa_nogcs
          GROUP BY icustay_id, starttime, endtime
          ORDER BY icustay_id, starttime   
        )

       -- We create the final table which includes length of stay, ventilation flag, rolling stay lengths etc
       , final AS
        (
         SELECT pat.subject_id, pat.hadm_id, merged.icustay_id
                , admission_type, admittime, dischtime, hospital_expire_flag, deathtime
                , intime, outtime
                , (EXTRACT(EPOCH FROM intime) - EXTRACT(EPOCH FROM admittime))/3600 AS admit_diff
                , CASE WHEN age <90 THEN age ELSE 90 END AS age
                , gender, ethnicity, insurance, initial_diagnosis, first_careunit, last_careunit, dbsource
                , los_hospital, hospstay_seq, los_icu, icustay_seq
                , merged.charttime
                , heart_rate, heart_rhythm
                , nbp_sys, nbp_dias, nbp_mean, abp_sys, abp_dias, abp_mean
                , temp_site, temp_celcius
                , verbal_response
                , o2sat, resp_rate, resp_rate_spont, resp_rate_set, resp_rate_total
                , minute_vol, mean_airway_pressure, peak_insp_rate, plateau_pressure
                , o2flow_chart, o2_device, vent_mode
                , case when ventdurations.icustay_id IS NULL THEN 0 ELSE 1 END AS on_vent
                , tidal_vol_set, tidal_vol_obs, tidal_vol_spon
                , peep_set, fio2_chart, glucose_chart
                , specimen, baseexcess, bicarbonate_bg, totalco2, carboxyhemoglobin, chloride_bg
                , calcium_bg, glucose_bg, hematocrit_bg, hemoglobin_bg, intubated, lactate
                , methemoglobin, o2flow_lab, fio2, so2, pco2, peep, ph, po2, potassium_bg
                , requiredo2, sodium_bg, temperature_bg, tidalvolume, ventilator 
                , alkalinephos, ast, bilirubin_direct, bilirubin_total, bun, creatinine
                , fibrinogen, magnesium, phosphate, platelets, ptt, tropinin_t, tropinin_i, wbc
                , bicarbonate, chloride, calcium, glucose, hematocrit, hemoglobin, potassium, sodium
                , (EXTRACT(EPOCH FROM merged.charttime) - EXTRACT(EPOCH FROM admittime))/3600/24 
                    AS rolling_los_hospital
                , (EXTRACT(EPOCH FROM merged.charttime) - EXTRACT(EPOCH FROM intime))/3600/24 
                    AS rolling_los_icu
                , respiration_24hours AS sofa_resp, coagulation_24hours AS sofa_coag
                , liver_24hours AS sofa_liver, cardiovascular_24hours AS sofa_circu
                , cns_24hours AS sofa_cns, renal_24hours AS sofa_renal
                , sofa_24hours AS sofa_total
           FROM patient_info pat
          INNER JOIN merged
            ON pat.icustay_id=merged.icustay_id
          LEFT JOIN ventdurations 
            ON merged.icustay_id = ventdurations.icustay_id 
           AND merged.charttime >= ventdurations.starttime
           AND merged.charttime <= ventdurations.endtime
         LEFT JOIN sofa_scores sofa
           ON merged.icustay_id = sofa.icustay_id
          AND merged.charttime >= sofa.starttime
          AND merged.charttime < sofa.endtime
        WHERE age>1 
        )

SELECT * FROM final
