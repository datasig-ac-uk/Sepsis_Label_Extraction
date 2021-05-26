/* 
This is a modified version of the one in mimic-lcp repo as we want to forward fill based on the last information we have. We create these other tables to find the last value of creatinine, bilirubin and platelets
There are benefits and limitations to either approach. We have chosen this as this view is used to get pivoted sofa scores and forward filling will produce fewer fluctuations in the SOFA score. The downside may be that older data is not always an accurate representation

We use the same assignment of icustay ids as we used in extracted_labs

I have left some comments written by Alistair and also commented a few additional lines out with justification. Note the blood gases were already commented out in the original version in mimic-lcp.

*/

DROP MATERIALIZED VIEW IF EXISTS pivoted_lab_ffill CASCADE;
CREATE MATERIALIZED VIEW pivoted_lab_ffill AS

        -- create a table which has fuzzy boundaries on ICU admission (+- 12 hours from documented time)
        -- this is used to assign icustay_id to lab data, which can be collected outside ICU
        -- involves first creating a lag/lead version of intime/outtime
        -- create a table which has fuzzy boundaries on ICU admission
        -- involves first creating a lag/lead version of intime/outtime
  WITH le AS
       (
          -- begin query that extracts the data
        SELECT subject_id, charttime
          -- here we assign labels to ITEMIDs
          -- this also fuses together multiple ITEMIDs containing the same data
               , CASE
                      WHEN itemid = 50868 THEN 'ANION GAP'
                      WHEN itemid = 50862 THEN 'ALBUMIN'
                      WHEN itemid = 51144 THEN 'BANDS'
                      WHEN itemid = 50882 THEN 'BICARBONATE'
                      WHEN itemid = 50885 THEN 'BILIRUBIN'
                      WHEN itemid = 50912 THEN 'CREATININE'
                      -- exclude blood gas
                      -- WHEN itemid = 50806 THEN 'CHLORIDE'
                      WHEN itemid = 50902 THEN 'CHLORIDE'
                      -- exclude blood gas
                      -- WHEN itemid = 50809 THEN 'GLUCOSE'
                      WHEN itemid = 50931 THEN 'GLUCOSE'
                      -- exclude blood gas
                      --WHEN itemid = 50810 THEN 'HEMATOCRIT'
                      WHEN itemid = 51221 THEN 'HEMATOCRIT'
                      -- exclude blood gas
                      --WHEN itemid = 50811 THEN 'HEMOGLOBIN'
                      WHEN itemid = 51222 THEN 'HEMOGLOBIN'
                      WHEN itemid = 50813 THEN 'LACTATE'
                      WHEN itemid = 51265 THEN 'PLATELET'
                      -- exclude blood gas
                      -- WHEN itemid = 50822 THEN 'POTASSIUM'
                      WHEN itemid = 50971 THEN 'POTASSIUM'
                      WHEN itemid = 51275 THEN 'PTT'
                      WHEN itemid = 51237 THEN 'INR'
                      WHEN itemid = 51274 THEN 'PT'
                      -- exclude blood gas
                      -- WHEN itemid = 50824 THEN 'SODIUM'
                      WHEN itemid = 50983 THEN 'SODIUM'
                      WHEN itemid = 51006 THEN 'BUN'
                      WHEN itemid = 51300 THEN 'WBC'
                      WHEN itemid = 51301 THEN 'WBC'
                 ELSE NULL
                  END AS label
               -- add in some sanity checks on the values
               -- the where clause below requires all valuenum to be > 0, so these are only upper limit checks
               , CASE
                      WHEN itemid = 50862 AND valuenum >  7    THEN NULL -- g/dL 'ALBUMIN'
                      WHEN itemid = 50862 AND valuenum <  0    THEN NULL -- g/dL 'ALBUMIN'
                      WHEN itemid = 50868 AND valuenum >  60   THEN NULL -- mEq/L 'ANION GAP' 
                      WHEN itemid = 50868 AND valuenum <  0    THEN NULL -- this can be negative in extreme cases, though that may also be caused by https://www.ajkd.org/article/S0272-6386(15)01055-0/pdf, though all negative values here seemed anomalous
                      WHEN itemid = 51144 AND valuenum <  0    THEN NULL -- immature band forms, %
                      WHEN itemid = 51144 AND valuenum >  100  THEN NULL -- immature band forms, %
                      WHEN itemid = 50882 AND valuenum >  60   THEN NULL -- mEq/L 'BICARBONATE'
                      WHEN itemid = 50882 AND valuenum <  0    THEN NULL -- mEq/L 'BICARBONATE'
                      WHEN itemid = 50885 AND valuenum >  83   THEN NULL -- mg/dL 'BILIRUBIN'
                      WHEN itemid = 50885 AND valuenum <  0    THEN NULL -- mg/dL 'BILIRUBIN'
                    -- WHEN itemid = 50806 and valuenum > 10000 THEN null -- mEq/L 'CHLORIDE'
                      WHEN itemid = 50902 AND valuenum >  160  THEN NULL -- mEq/L 'CHLORIDE'
                      WHEN itemid = 50902 AND valuenum <= 50   THEN NULL -- mEq/L 'CHLORIDE'
                      WHEN itemid = 50912 AND valuenum >  45   THEN NULL -- mg/dL 'CREATININE'
                      WHEN itemid = 50912 AND valuenum <= 0    THEN NULL -- mg/dL 'CREATININE'
                    --WHEN itemid = 50809 and valuenum > 10000 THEN null -- mg/dL 'GLUCOSE'
                      WHEN itemid = 50931 AND valuenum >  1300 THEN NULL -- mg/dL 'GLUCOSE'
                      WHEN itemid = 50931 AND valuenum <  0    THEN NULL -- mg/dL 'GLUCOSE'
                   -- WHEN itemid = 50810 and valuenum >   100 THEN null -- % 'HEMATOCRIT'
                      WHEN itemid = 51221 AND valuenum >  100  THEN NULL -- % 'HEMATOCRIT'
                      WHEN itemid = 51221 AND valuenum <  0    THEN NULL -- % 'HEMATOCRIT'
                  --  WHEN itemid = 50811 and valuenum >    50 THEN null -- g/dL 'HEMOGLOBIN'
                      WHEN itemid = 51222 AND valuenum >  30   THEN NULL -- g/dL 'HEMOGLOBIN'
                      WHEN itemid = 51222 AND valuenum <= 0    THEN NULL -- g/dL 'HEMOGLOBIN'
                      WHEN itemid = 50813 AND valuenum >  40   THEN NULL -- mmol/L 'LACTATE'
                      WHEN itemid = 50813 AND valuenum <  0    THEN NULL -- mmol/L 'LACTATE'
                      WHEN itemid = 51265 AND valuenum >  2000 THEN NULL -- K/uL 'PLATELET'
                      WHEN itemid = 51265 AND valuenum <  0    THEN NULL -- K/uL 'PLATELET'
                  --  WHEN itemid = 50822 and valuenum >  20 THEN null -- mEq/L 'POTASSIUM'
                      WHEN itemid = 50971 AND valuenum >  12   THEN NULL -- mEq/L 'POTASSIUM'
                      WHEN itemid = 50971 AND valuenum <  1.4  THEN NULL -- mEq/L 'POTASSIUM'
                      WHEN itemid = 51275 AND valuenum >  150  THEN NULL -- sec 'PTT'
                      WHEN itemid = 51275 AND valuenum <  0    THEN NULL -- sec 'PTT'
                      WHEN itemid = 51275 AND value in ('>150', '150 IS HIGHEST MEASURED PTT') then 160 -- sec 'PTT'
                      WHEN itemid = 51237 AND valuenum >  90   THEN NULL -- 'INR'
                      WHEN itemid = 51237 AND valuenum <  0    THEN NULL -- 'INR'
                      WHEN itemid = 51274 AND valuenum >  150  THEN NULL -- sec 'PT'
                      WHEN itemid = 51274 AND valuenum <  0    THEN NULL -- sec 'PT'
                  --  WHEN itemid = 50824 and valuenum >  185 THEN null -- mEq/L == mmol/L 'SODIUM'
                      WHEN itemid = 50983 AND valuenum >  185  THEN NULL -- mEq/L == mmol/L 'SODIUM'
                      WHEN itemid = 50983 AND valuenum <  80   THEN NULL -- mEq/L == mmol/L 'SODIUM'
                      WHEN itemid = 51006 AND valuenum >  300  THEN NULL -- 'BUN'
                      WHEN itemid = 51006 AND valuenum <  0    THEN NULL -- 'BUN'
                      WHEN itemid = 51300 AND valuenum >  700  THEN NULL -- 'WBC'
                      WHEN itemid = 51300 AND valuenum <  0    THEN NULL -- 'WBC'
                      WHEN itemid = 51300 AND value = '<0.1'   THEN 0.05 -- 'WBC'
                      WHEN itemid = 51301 AND valuenum >  700  THEN NULL -- 'WBC'
                      WHEN itemid = 51301 AND valuenum <  0    THEN NULL -- 'WBC'
                      WHEN itemid = 51301 AND value = '<0.1'   THEN 0.05 -- 'WBC'
                 ELSE valuenum
                  END AS valuenum
          FROM labevents
         WHERE ITEMID IN
              (
                -- comment is: LABEL | CATEGORY | FLUID | NUMBER OF ROWS IN LABEVENTS
                50868, -- ANION GAP | CHEMISTRY | BLOOD | 769895
                50862, -- ALBUMIN | CHEMISTRY | BLOOD | 146697
                51144, -- BANDS - hematology
                50882, -- BICARBONATE | CHEMISTRY | BLOOD | 780733
                50885, -- BILIRUBIN, TOTAL | CHEMISTRY | BLOOD | 238277
                50912, -- CREATININE | CHEMISTRY | BLOOD | 797476
                50902, -- CHLORIDE | CHEMISTRY | BLOOD | 795568
                -- 50806, -- CHLORIDE, WHOLE BLOOD | BLOOD GAS | BLOOD | 48187
                50931, -- GLUCOSE | CHEMISTRY | BLOOD | 748981
                -- 50809, -- GLUCOSE | BLOOD GAS | BLOOD | 196734
                51221, -- HEMATOCRIT | HEMATOLOGY | BLOOD | 881846
                -- 50810, -- HEMATOCRIT, CALCULATED | BLOOD GAS | BLOOD | 89715
                51222, -- HEMOGLOBIN | HEMATOLOGY | BLOOD | 752523
                -- 50811, -- HEMOGLOBIN | BLOOD GAS | BLOOD | 89712
                50813, -- LACTATE | BLOOD GAS | BLOOD | 187124
                51265, -- PLATELET COUNT | HEMATOLOGY | BLOOD | 778444
                50971, -- POTASSIUM | CHEMISTRY | BLOOD | 845825
                -- 50822, -- POTASSIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 192946
                51275, -- PTT | HEMATOLOGY | BLOOD | 474937
                51237, -- INR(PT) | HEMATOLOGY | BLOOD | 471183
                51274, -- PT | HEMATOLOGY | BLOOD | 469090
                50983, -- SODIUM | CHEMISTRY | BLOOD | 808489
                -- 50824, -- SODIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 71503
                51006, -- UREA NITROGEN | CHEMISTRY | BLOOD | 791925
                51301, -- WHITE BLOOD CELLS | HEMATOLOGY | BLOOD | 753301
                51300  -- WBC COUNT | HEMATOLOGY | BLOOD | 2371
              )
            --  AND valuenum IS NOT NULL AND valuenum > 0 -- lab values cannot be 0 and cannot be negative
            -- I got rid of the above line compared with mimic-lcp as we are keeping cases where the valuenum is null but the value may give us information as a text entry
        )
       , le_avg AS
        (
         SELECT le.subject_id, le.charttime
                , AVG(CASE WHEN label = 'ANION GAP' THEN valuenum ELSE null END) AS ANIONGAP
                , AVG(CASE WHEN label = 'ALBUMIN' THEN valuenum ELSE null END) AS ALBUMIN
                , AVG(CASE WHEN label = 'BANDS' THEN valuenum ELSE null END) AS BANDS
                , AVG(CASE WHEN label = 'BICARBONATE' THEN valuenum ELSE null END) AS BICARBONATE
                , AVG(CASE WHEN label = 'BILIRUBIN' THEN valuenum ELSE null END) AS BILIRUBIN
                , AVG(CASE WHEN label = 'CREATININE' THEN valuenum ELSE null END) AS CREATININE
                , AVG(CASE WHEN label = 'CHLORIDE' THEN valuenum ELSE null END) AS CHLORIDE
                , AVG(CASE WHEN label = 'GLUCOSE' THEN valuenum ELSE null END) AS GLUCOSE
                , AVG(CASE WHEN label = 'HEMATOCRIT' THEN valuenum ELSE null END) AS HEMATOCRIT
                , AVG(CASE WHEN label = 'HEMOGLOBIN' THEN valuenum ELSE null END) AS HEMOGLOBIN
                , AVG(CASE WHEN label = 'LACTATE' THEN valuenum ELSE null END) AS LACTATE
                , AVG(CASE WHEN label = 'PLATELET' THEN valuenum ELSE null END) AS PLATELET
                , AVG(CASE WHEN label = 'POTASSIUM' THEN valuenum ELSE null END) AS POTASSIUM
                , AVG(CASE WHEN label = 'PTT' THEN valuenum ELSE null END) AS PTT
                , AVG(CASE WHEN label = 'INR' THEN valuenum ELSE null END) AS INR
                , AVG(CASE WHEN label = 'PT' THEN valuenum ELSE null END) AS PT
                , AVG(CASE WHEN label = 'SODIUM' THEN valuenum ELSE null end) AS SODIUM
                , AVG(CASE WHEN label = 'BUN' THEN valuenum ELSE null end) AS BUN
                , AVG(CASE WHEN label = 'WBC' THEN valuenum ELSE null end) AS WBC
           FROM le
          GROUP BY le.subject_id, le.charttime
        )

        -- assign an icustay_id
       , all_data AS
       (
        SELECT id.icustay_id, id.hadm_id, le_avg.*
          FROM le_avg
          LEFT JOIN assign_lab_ids id
            ON le_avg.subject_id  = id.subject_id
           AND le_avg.charttime >= id.starttime
           AND le_avg.charttime  < id.endtime
         ORDER BY le_avg.subject_id, le_avg.charttime;
        )
    
        -- Next we create some more tables to forward fill the data since lab data is so sparse
       , creatinine AS
       (
        SELECT lab.icustay_id, creatinine, charttime AS starttime
               , LEAD (charttime) OVER (PARTITION BY lab.icustay_id ORDER BY charttime) AS charttime_lead
               , outtime
          FROM all_data lab
          LEFT JOIN icustays ic ON lab.icustay_id = ic.icustay_id
         WHERE creatinine IS NOT NULL
         ORDER BY icustay_id, charttime
        )

       , creatinine_times AS
       (
        SELECT * 
            --define end valid time of this lab measurement
               , COALESCE(charttime_lead, GREATEST(outtime, starttime + INTERVAL '48' HOUR)) AS endtime 
          FROM creatinine
        )

       , bilirubin AS
       (
        SELECT lab.icustay_id, bilirubin, charttime AS starttime
               , lead (charttime) OVER (PARTITION BY lab.icustay_id ORDER BY charttime) AS charttime_lead
               , outtime
          FROM all_data lab
          LEFT JOIN icustays ic ON lab.icustay_id = ic.icustay_id
         WHERE bilirubin IS NOT NULL
         ORDER BY icustay_id, charttime
        )
        
       , bilirubin_times AS
       (
        SELECT * 
            --define end valid time of this lab measurement
               , COALESCE(charttime_lead, greatest(outtime, starttime + INTERVAL '48' HOUR)) AS endtime 
          FROM bilirubin
        )

       , platelet AS
       (
        SELECT lab.icustay_id, platelet, charttime AS starttime
               , LEAD (charttime) OVER (PARTITION BY lab.icustay_id ORDER BY charttime) AS charttime_lead
               , outtime
          FROM all_data lab
          LEFT JOIN icustays ic ON lab.icustay_id = ic.icustay_id
         WHERE platelet IS NOT NULL
         ORDER BY icustay_id, charttime
        )

       , platelet_times AS
       (
        SELECT * 
                --define end valid time of this lab measurement
               , COALESCE(charttime_lead, greatest(outtime, starttime + INTERVAL '48' HOUR)) AS endtime 
          FROM platelet
        )

SELECT lab.*, cr.creatinine AS last_creatinine, bi.bilirubin AS last_bilirubin
       , pl.platelet AS last_platelet
  FROM all_data lab
  LEFT JOIN creatinine_times cr
    ON lab.icustay_id = cr.icustay_id
   AND lab.charttime>= cr.starttime
   AND lab.charttime < cr.endtime
  LEFT JOIN bilirubin_times bi
    ON lab.icustay_id = bi.icustay_id
   AND lab.charttime>= bi.starttime
   AND lab.charttime < bi.endtime
  LEFT JOIN platelet_times pl
    ON lab.icustay_id = pl.icustay_id
   AND lab.charttime>= pl.starttime
   AND lab.charttime < pl.endtime
 ORDER BY icustay_id, charttime;