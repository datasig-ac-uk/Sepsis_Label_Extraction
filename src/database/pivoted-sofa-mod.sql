/* 
This is also adapted from the version in the repository MIT-LCP, written by Alistair Johnson. We use a couple of amended tables here.

We are using forward filled values for the labs.

We change the boundary so that the start time is inclusive and the end time is exclusive instead, it feels more intuitive sense. As a result we have also changed from the endtime to starttime for checking vasopressors

We also modified the definition of the ventilation component slightly to include the case where a patient is on ventilation but has a PF of above 200.

The following is a description of SOFA by Alistair (with a couple of changes to the list of views).
*/

-- ------------------------------------------------------------------
-- Title: Sequential Organ Failure Assessment (SOFA)
-- This query extracts the sequential organ failure assessment (formally: sepsis-related organ failure assessment).
-- This score is a measure of organ failure for patients in the ICU.
-- The score is calculated for every hour of the patient's ICU stay.
-- However, as the calculation window is 24 hours, care should be taken when
-- using the score before the end of the first day.
-- ------------------------------------------------------------------

-- Reference for SOFA:
--    Jean-Louis Vincent, Rui Moreno, Jukka Takala, Sheila Willatts, Arnaldo De Mendonça,
--    Hajo Bruining, C. K. Reinhart, Peter M Suter, and L. G. Thijs.
--    "The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure."
--    Intensive care medicine 22, no. 7 (1996): 707-710.

-- Variables used in SOFA:
--  GCS, MAP, FiO2, Ventilation status (sourced from CHARTEVENTS)
--  Creatinine, Bilirubin, FiO2, PaO2, Platelets (sourced from LABEVENTS)
--  Dopamine, Dobutamine, Epinephrine, Norepinephrine (sourced from INPUTEVENTS_MV and INPUTEVENTS_CV)
--  Urine output (sourced from OUTPUTEVENTS)

-- The following views required to run this query:
--  1) pivoted_bg_art - generated by pivoted-bg.sql
--  2) pivoted_uo - generated by pivoted-uo.sql
--  3) pivoted_lab - generated by pivoted-lab.sql
--  4) pivoted_gcs - generated by pivoted-gcs.sql
--  5) ventdurations - generated by ../durations/ventilation-durations.sql
--  6) norepinephrine_dose - generated by ../durations/norepinephrine-dose.sql
--  7) epinephrine_dose - generated by ../durations/epinephrine-dose.sql
--  8) dopamine_dose - generated by ../durations/dopamine-dose.sql
--  9) dobutamine_dose - generated by ../durations/dobutamine-dose.sql

-- Note:
--  The score is calculated for only adult ICU patients,

  DROP TABLE IF EXISTS pivoted_sofa_mod CASCADE;
CREATE TABLE pivoted_sofa_mod AS

    -- generate a row for every hour the patient was in the ICU
   
  WITH icu_time AS
       (
        SELECT i.icustay_id, i.hr AS in_hr, o.hr AS out_hr
          FROM 
               (
                SELECT times.icustay_id, hr
                  FROM times 
                 INNER JOIN icustays
                    ON times.icustay_id = icustays.icustay_id
                   AND times.starttime <= icustays.intime
                   AND times.endtime > icustays.intime
                ) i
         INNER JOIN    
               (
                SELECT times.icustay_id, hr
                  FROM times 
                 INNER JOIN icustays
                    ON times.icustay_id = icustays.icustay_id
                   AND times.starttime <= icustays.outtime
                   AND times.endtime > icustays.outtime
                ) o
            ON i.icustay_id = o.icustay_id
        )

         -- get minimum blood pressure from chartevents
       , bp AS
       (
        SELECT ce.icustay_id
               , ce.charttime
               , MIN(valuenum) AS MeanBP_min
          FROM chartevents ce
          -- exclude rows marked as error
         WHERE ce.error IS DISTINCT FROM 1
           AND ce.itemid IN
                  (
                  -- MEAN ARTERIAL PRESSURE
                  456, --"NBP Mean"
                  52, --"Arterial BP Mean"
                  6702, -- Arterial BP Mean #2
                  443, --   Manual BP Mean(calc)
                  220052, --"Arterial Blood Pressure mean"
                  220181, --"Non Invasive Blood Pressure mean"
                  225312  --"ART BP mean"
                  )
           AND valuenum > 0 AND valuenum < 300
         GROUP BY ce.icustay_id, ce.charttime
        )
        
       , mini_agg AS
       (
        SELECT co.icustay_id, co.hr
               -- vitals
               , MIN(bp.MeanBP_min) AS MeanBP_min
               -- gcs
               , MIN(gcs.GCS) AS GCS_min
               -- labs
               , MAX(labs.last_bilirubin) AS bilirubin_max
               , MAX(labs.last_creatinine) AS creatinine_max
               , MIN(labs.last_platelet) AS platelet_min
               -- because pafi has an interaction between vent/PaO2:FiO2, we need two columns for the score
               -- it can happen that the lowest unventilated PaO2/FiO2 is 68, but the lowest ventilated PaO2/FiO2 is 120
               -- in this case, the SOFA score is 3, *not* 4.
               , MIN(CASE WHEN vd.icustay_id IS NULL THEN pao2fio2ratio ELSE NULL END) AS PaO2FiO2Ratio_novent
               , MIN(CASE WHEN vd.icustay_id IS NOT NULL THEN pao2fio2ratio ELSE NULL END) AS PaO2FiO2Ratio_vent
          FROM times co
          LEFT JOIN bp
            ON co.icustay_id = bp.icustay_id
           AND co.starttime <= bp.charttime
           AND co.endtime > bp.charttime
          LEFT JOIN pivoted_gcs gcs
            ON co.icustay_id = gcs.icustay_id
           AND co.starttime <= gcs.charttime
           AND co.endtime > gcs.charttime
          LEFT JOIN pivoted_lab_ffill labs
            ON co.hadm_id = labs.hadm_id
           AND co.starttime <= labs.charttime
           AND co.endtime > labs.charttime
          -- bring in blood gases that occurred during this hour
          LEFT JOIN pivoted_bg_art_mod bg
            ON co.icustay_id = bg.icustay_id
           AND co.starttime <= bg.charttime
           AND co.endtime > bg.charttime
          -- at the time of the blood gas, determine if patient was ventilated
          LEFT JOIN ventdurations vd
            ON co.icustay_id = vd.icustay_id
           AND bg.charttime >= vd.starttime
           AND bg.charttime <= vd.endtime
         GROUP BY co.icustay_id, co.hr
        )
        
        -- sum uo separately to prevent duplicating values
       , uo AS
       (
        SELECT co.icustay_id, co.hr
          -- uo
               , sum(uo.urineoutput) AS UrineOutput
          FROM times co
          LEFT JOIN pivoted_uo uo
            ON co.icustay_id = uo.icustay_id
           AND co.starttime <= uo.charttime
           AND co.endtime > uo.charttime
         GROUP BY co.icustay_id, co.hr
        )
        
       , scorecomp AS
       (
        SELECT co.icustay_id
               , co.hr
               , icu.in_hr
               , icu.out_hr
               , co.starttime, co.endtime
               , ma.PaO2FiO2Ratio_novent
               , ma.PaO2FiO2Ratio_vent
               , epi.vaso_rate AS rate_epinephrine
               , nor.vaso_rate AS rate_norepinephrine
               , dop.vaso_rate AS rate_dopamine
               , dob.vaso_rate AS rate_dobutamine
               , ma.MeanBP_min
               , ma.GCS_min
                -- uo
               , uo.urineoutput
                -- labs
               , ma.bilirubin_max
               , ma.creatinine_max
               , ma.platelet_min
           
          FROM times co
          LEFT JOIN icu_time icu
            ON co.icustay_id = icu.icustay_id
          LEFT JOIN mini_agg ma
            ON co.icustay_id = ma.icustay_id
           AND co.hr = ma.hr
          LEFT JOIN uo 
            ON co.icustay_id = uo.icustay_id
           AND co.hr = uo.hr
          -- add in dose of vasopressors
          -- dose tables have 1 row for each start/stop interval,
          -- so no aggregation needed
          LEFT JOIN epinephrine_dose epi
            ON co.icustay_id = epi.icustay_id
           AND co.starttime > epi.starttime
           AND co.starttime <= epi.endtime
          LEFT JOIN norepinephrine_dose nor
            ON co.icustay_id = nor.icustay_id
           AND co.starttime > nor.starttime
           AND co.starttime <= nor.endtime
          LEFT JOIN dopamine_dose dop
            ON co.icustay_id = dop.icustay_id
           AND co.starttime > dop.starttime
           AND co.starttime <= dop.endtime
          LEFT JOIN dobutamine_dose dob
            ON co.icustay_id = dob.icustay_id
           AND co.starttime > dob.starttime
           AND co.starttime <= dob.endtime
        )
        
       , scorecalc AS
       (
          -- Calculate the final score
          -- note that if the underlying data is missing, the component is null
          -- eventually these are treated as 0 (normal), but knowing when data is missing is useful for debugging. 
          -- Note again we are using the forward filled version of the labs
        
        SELECT scorecomp.*
                  -- Respiration
               , CASE
                -- a minor change from the original to include patients on ventilation but above 200
                      WHEN PaO2FiO2Ratio_vent   < 100 THEN 4 --OR PaO2FiO2Ratio_novent < 100 then 4
                      WHEN PaO2FiO2Ratio_vent   < 200 THEN 3 --OR PaO2FiO2Ratio_novent < 200 then 3
                      WHEN PaO2FiO2Ratio_novent < 300 
                           OR PaO2FiO2Ratio_vent < 300 THEN 2
                      WHEN PaO2FiO2Ratio_novent < 400 
                           OR PaO2FiO2Ratio_vent < 400 THEN 1
                      WHEN coalesce(PaO2FiO2Ratio_vent, PaO2FiO2Ratio_novent) IS NULL THEN NULL
                      ELSE 0
                   END::SMALLINT 
                    AS respiration

              -- Coagulation
               , CASE
                      WHEN platelet_min < 20  THEN 4
                      WHEN platelet_min < 50  THEN 3
                      WHEN platelet_min < 100 THEN 2
                      WHEN platelet_min < 150 THEN 1
                      WHEN platelet_min IS NULL THEN NULL
                      ELSE 0
                  END::SMALLINT 
                   AS coagulation

              -- Liver
               , CASE
                      -- Bilirubin checks in mg/dL
                      WHEN Bilirubin_Max >= 12.0 THEN 4
                      WHEN Bilirubin_Max >= 6.0  THEN 3
                      WHEN Bilirubin_Max >= 2.0  THEN 2
                      WHEN Bilirubin_Max >= 1.2  THEN 1
                      WHEN Bilirubin_Max IS NULL THEN NULL
                      ELSE 0
                  END::SMALLINT 
                   AS liver

              -- Cardiovascular
               , CASE
                      WHEN rate_dopamine > 15 
                           OR rate_epinephrine >  0.1 
                           OR rate_norepinephrine >  0.1 
                      THEN 4
                      WHEN rate_dopamine >  5 
                           OR rate_epinephrine <= 0.1 
                           OR rate_norepinephrine <= 0.1 
                      THEN 3
                      WHEN rate_dopamine >  0 
                           OR rate_dobutamine > 0 
                      THEN 2
                      WHEN MeanBP_Min < 70 
                      THEN 1
                      WHEN COALESCE(MeanBP_Min, rate_dopamine, rate_dobutamine, rate_epinephrine, rate_norepinephrine) IS NULL 
                      THEN NULL
                      ELSE 0
                  END::SMALLINT 
                   AS cardiovascular

              -- Neurological failure (GCS)
               , CASE
                      WHEN (GCS_min >= 13 AND GCS_min <= 14) THEN 1
                      WHEN (GCS_min >= 10 AND GCS_min <= 12) THEN 2
                      WHEN (GCS_min >=  6 AND GCS_min <=  9) THEN 3
                      WHEN  GCS_min <   6 THEN 4
                      WHEN  GCS_min IS NULL THEN NULL
                      ELSE 0 
                  END::SMALLINT 
                   AS cns

               -- Renal failure - high creatinine or low urine output
               , CASE
                      WHEN (Creatinine_Max >= 5.0) 
                      THEN 4
                      WHEN SUM(urineoutput) OVER W < 200 
                           AND hr >= in_hr + 24 
                           AND hr <= out_hr
                      THEN 4
                      WHEN (Creatinine_Max >= 3.5 
                           AND Creatinine_Max < 5.0) 
                      THEN 3
                      WHEN SUM(urineoutput) OVER W < 500 
                           AND hr >= in_hr+24 
                           AND hr <= out_hr
                      THEN 3
                      WHEN (Creatinine_Max >= 2.0 
                           AND Creatinine_Max < 3.5) 
                      THEN 2
                      WHEN (Creatinine_Max >= 1.2 
                           AND Creatinine_Max < 2.0) 
                      THEN 1
                      WHEN COALESCE
                                  (
                                    SUM(urineoutput) OVER W
                                    , Creatinine_Max
                                  ) IS NULL THEN NULL
                      ELSE 0 
                  END::SMALLINT 
                   AS renal
          FROM scorecomp
        WINDOW W AS
                    (
                     PARTITION BY icustay_id
                     ORDER BY hr
                     ROWS BETWEEN 23 PRECEDING 
                              AND 0 FOLLOWING
                    )
        )
        
       , score_final AS
       (
        SELECT s.*
               -- Combine all the scores to get SOFA
               -- Impute 0 if the score is missing
               -- the window function takes the max over the last 24 hours
               , COALESCE(MAX(respiration) OVER W, 0)::SMALLINT AS respiration_24hours
               , COALESCE(MAX(coagulation) OVER W, 0)::SMALLINT AS coagulation_24hours
               , COALESCE(MAX(liver) OVER W, 0)::SMALLINT AS liver_24hours
               , COALESCE(MAX(cardiovascular) OVER W,0)::SMALLINT AS cardiovascular_24hours
               , COALESCE(MAX(cns) OVER W,0)::SMALLINT AS cns_24hours
               , COALESCE(MAX(renal) OVER W,0)::SMALLINT AS renal_24hours

               -- sum together data for final SOFA
               , (COALESCE(MAX(respiration) OVER W,0)
                     + COALESCE(MAX(coagulation) OVER W, 0)
                     + COALESCE(MAX(liver) OVER W, 0)
                     + COALESCE(MAX(cardiovascular) OVER W, 0)
                     + COALESCE(MAX(cns) OVER W, 0)
                     + COALESCE(MAX(renal) OVER W, 0)
                  )::SMALLINT
                 AS SOFA_24hours

          FROM scorecalc s
        WINDOW W AS
               (
                PARTITION BY icustay_id
                ORDER BY hr
                ROWS BETWEEN 23 PRECEDING 
                         AND 0 FOLLOWING
               )
        )

SELECT * 
  FROM score_final
  -- WHERE hr >= 0
  -- We have commented this out as we want to look at some data in the ward (limited labs) in the ward too
 ORDER BY icustay_id, hr;
