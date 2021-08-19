/* 
The aim of this query is to pivot entries related to blood gases and chemistry values which were found in LABEVENTS.

The main change from MIT-LCP is that we have introduced more/tighter error bounds, motivated by the fact that base excess can be negative. 

The assignment of icustay_ids match our other queries.

The definition of stg_fio2 is modified to include some other codes
*/

  DROP MATERIALIZED VIEW IF EXISTS pivoted_bg_mod CASCADE;
CREATE MATERIALIZED VIEW pivoted_bg_mod AS

        -- create a table which has fuzzy boundaries on ICU admission
        -- involves first creating a lag/lead version of intime/outtime
  WITH pvt AS
       ( -- begin query that extracts the data
        SELECT le.subject_id
          -- here we assign labels to ITEMIDs
          -- this also fuses together multiple ITEMIDs containing the same data
               , CASE
                      WHEN itemid = 50800 THEN 'SPECIMEN'
                      WHEN itemid = 50801 THEN 'AADO2'
                      WHEN itemid = 50802 THEN 'BASEEXCESS'
                      WHEN itemid = 50803 THEN 'BICARBONATE'
                      WHEN itemid = 50804 THEN 'TOTALCO2'
                      WHEN itemid = 50805 THEN 'CARBOXYHEMOGLOBIN'
                      WHEN itemid = 50806 THEN 'CHLORIDE'
                      WHEN itemid = 50808 THEN 'CALCIUM'
                      WHEN itemid = 50809 THEN 'GLUCOSE'
                      WHEN itemid = 50810 THEN 'HEMATOCRIT'
                      WHEN itemid = 50811 THEN 'HEMOGLOBIN'
                      WHEN itemid = 50812 THEN 'INTUBATED'
                      WHEN itemid = 50813 THEN 'LACTATE'
                      WHEN itemid = 50814 THEN 'METHEMOGLOBIN'
                      WHEN itemid = 50815 THEN 'O2FLOW'
                      WHEN itemid = 50816 THEN 'FIO2'
                      WHEN itemid = 50817 THEN 'SO2' 
                      WHEN itemid = 50818 THEN 'PCO2'
                      WHEN itemid = 50819 THEN 'PEEP'
                      WHEN itemid = 50820 THEN 'PH'
                      WHEN itemid = 50821 THEN 'PO2'
                      WHEN itemid = 50822 THEN 'POTASSIUM'
                      WHEN itemid = 50823 THEN 'REQUIREDO2'
                      WHEN itemid = 50824 THEN 'SODIUM'
                      WHEN itemid = 50825 THEN 'TEMPERATURE'
                      WHEN itemid = 50826 THEN 'TIDALVOLUME'
                      WHEN itemid = 50827 THEN 'VENTILATIONRATE'
                      WHEN itemid = 50828 THEN 'VENTILATOR'
                 ELSE NULL
                  END AS label
               , charttime
               , value
                -- add in some sanity checks on the values
               , CASE
                --when valuenum < 0 then null 
                -- Got rid of the above as base excess can be negative!
                      WHEN itemid = 50801 AND valuenum > 800 THEN NULL
                      WHEN itemid = 50801 AND valuenum <= 0 THEN NULL
                      WHEN itemid = 50802 AND valuenum > 40 THEN NULL
                      WHEN itemid = 50802 AND valuenum < -40 THEN NULL
                      WHEN itemid = 50803 AND valuenum > 60 THEN NULL
                      WHEN itemid = 50803 AND valuenum < 0 THEN NULL
                      WHEN itemid = 50804 AND valuenum > 120 THEN NULL
                      WHEN itemid = 50804 AND valuenum < 0 THEN NULL
                      WHEN itemid = 50805 AND valuenum > 20 THEN NULL
                      WHEN itemid = 50805 AND valuenum < 0 THEN NULL
                      WHEN itemid = 50806 AND valuenum > 160 THEN NULL
                      WHEN itemid = 50806 AND valuenum < 50 THEN NULL
                      WHEN itemid = 50808 AND valuenum > 4 THEN NULL
                      WHEN itemid = 50808 AND valuenum <= 0 THEN NULL
                      WHEN itemid = 50809 AND valuenum > 1300 THEN NULL
                      WHEN itemid = 50809 AND valuenum <= 0 THEN NULL
                      WHEN itemid = 50810 AND valuenum > 80 THEN NULL
                      WHEN itemid = 50810 AND valuenum < 0 THEN NULL
                      WHEN itemid = 50811 AND valuenum > 30 THEN NULL
                      WHEN itemid = 50811 AND valuenum <= 0 THEN NULL
                      WHEN itemid = 50813 AND valuenum > 40 THEN NULL
                      WHEN itemid = 50813 AND valuenum < 0 THEN NULL
                      WHEN itemid = 50814 AND valuenum > 100 THEN NULL
                      WHEN itemid = 50814 AND valuenum < 0 THEN NULL
                      WHEN itemid = 50815 AND valuenum > 100 THEN NULL
                      WHEN itemid = 50815 AND valuenum < 0 THEN NULL
                      WHEN itemid = 50816 AND valuenum > 100 THEN NULL
                      WHEN itemid = 50816 AND valuenum < 20 THEN NULL
                      WHEN itemid = 50817 AND valuenum > 100 THEN NULL
                      WHEN itemid = 50817 AND valuenum < 0 THEN NULL
                      WHEN itemid = 50818 AND valuenum > 250 THEN NULL
                      WHEN itemid = 50818 AND valuenum < 0 THEN NULL
                      WHEN itemid = 50819 AND valuenum > 50 THEN NULL
                      WHEN itemid = 50819 AND valuenum < 0 THEN NULL
                      WHEN itemid = 50820 AND valuenum >= 8 THEN NULL
                      WHEN itemid = 50820 AND valuenum <= 6.2 THEN NULL
                      WHEN itemid = 50821 AND valuenum > 800 THEN NULL
                      WHEN itemid = 50821 AND valuenum < 0 THEN NULL
                      WHEN itemid = 50822 AND valuenum > 12 THEN NULL
                      WHEN itemid = 50822 AND valuenum < 1.4 THEN NULL
                      WHEN itemid = 50823 AND valuenum > 100 THEN NULL
                      WHEN itemid = 50823 AND valuenum < 20 THEN NULL
                      WHEN itemid = 50824 AND valuenum > 185 THEN NULL
                      WHEN itemid = 50824 AND valuenum < 80 THEN NULL
                      WHEN itemid = 50825 AND valuenum > 45 THEN NULL
                      WHEN itemid = 50825 AND valuenum < 15 THEN NULL
                      WHEN itemid = 50826 AND valuenum > 5000 THEN NULL
                      WHEN itemid = 50826 AND valuenum < 0 THEN NULL

                  -- The following are the list of original error bounds from MIT-LCP
                  -- when itemid = 50810 and valuenum > 100 then null 
                  -- ensure FiO2 is a valid number between 21-100
                  -- mistakes are rare (<100 obs out of ~100,000)
                  -- there are 862 obs of valuenum == 20 - some people round down!
                  -- rather than risk imputing garbage data for FiO2, we simply NULL invalid values
           --       when itemid = 50816 and valuenum < 20 then null
            --      when itemid = 50816 and valuenum > 100 then null
            ---      when itemid = 50817 and valuenum > 100 then null -- O2 sat
            --      when itemid = 50815 and valuenum >  70 then null -- O2 flow
            --      when itemid = 50821 and valuenum > 800 then null -- PO2
                   -- conservative upper limit
                 ELSE valuenum
                  END AS valuenum
          FROM labevents le
         WHERE le.ITEMID IN
                    -- blood gases
                    (
                      50800, 50801, 50802, 50803, 50804, 50805, 50806, 50807, 50808, 50809
                      , 50810, 50811, 50812, 50813, 50814, 50815, 50816, 50817, 50818, 50819
                      , 50820, 50821, 50822, 50823, 50824, 50825, 50826, 50827, 50828
                      , 51545
                    )
        )
        
        -- Now create a pivoted view
       , grp AS
       (
        SELECT pvt.subject_id, pvt.charttime
               , MAX(CASE WHEN label = 'SPECIMEN' THEN value ELSE NULL END) AS SPECIMEN
               , AVG(CASE WHEN label = 'AADO2' THEN valuenum ELSE NULL END) AS AADO2
               , AVG(CASE WHEN label = 'BASEEXCESS' THEN valuenum ELSE NULL END) AS BASEEXCESS
               , AVG(CASE WHEN label = 'BICARBONATE' THEN valuenum ELSE NULL END) AS BICARBONATE
               , AVG(CASE WHEN label = 'TOTALCO2' THEN valuenum ELSE NULL END) AS TOTALCO2
               , AVG(CASE WHEN label = 'CARBOXYHEMOGLOBIN' THEN valuenum ELSE NULL END) AS CARBOXYHEMOGLOBIN
               , AVG(CASE WHEN label = 'CHLORIDE' THEN valuenum ELSE NULL END) AS CHLORIDE
               , AVG(CASE WHEN label = 'CALCIUM' THEN valuenum ELSE NULL END) AS CALCIUM
               , AVG(CASE WHEN label = 'GLUCOSE' THEN valuenum ELSE NULL END) AS GLUCOSE
               , AVG(CASE WHEN label = 'HEMATOCRIT' THEN valuenum ELSE NULL END) AS HEMATOCRIT
               , AVG(CASE WHEN label = 'HEMOGLOBIN' THEN valuenum ELSE NULL END) AS HEMOGLOBIN
                -- changed intubated definition as there was no valuenum
               , MAX(CASE WHEN label = 'INTUBATED' AND value = 'INTUBATED' THEN 1 WHEN label = 'INTUBATED' AND value = 'NOT INTUBATED' THEN 0 ELSE NULL END) AS INTUBATED
               , AVG(CASE WHEN label = 'LACTATE' THEN valuenum ELSE NULL END) AS LACTATE
               , AVG(CASE WHEN label = 'METHEMOGLOBIN' then valuenum ELSE NULL END) AS METHEMOGLOBIN
               , AVG(CASE WHEN label = 'O2FLOW' THEN valuenum ELSE NULL END) AS O2FLOW
               , AVG(CASE WHEN label = 'FIO2' THEN valuenum ELSE NULL END) AS FIO2
               , AVG(CASE WHEN label = 'SO2' THEN valuenum ELSE NULL END) AS SO2 -- OXYGENSATURATION
               , AVG(CASE WHEN label = 'PCO2' THEN valuenum ELSE NULL END) AS PCO2
               , AVG(CASE WHEN label = 'PEEP' THEN valuenum ELSE NULL END) AS PEEP
               , AVG(CASE WHEN label = 'PH' THEN valuenum ELSE NULL END) AS PH
               , AVG(CASE WHEN label = 'PO2' THEN valuenum ELSE NULL END) AS PO2
               , AVG(CASE WHEN label = 'POTASSIUM' THEN valuenum ELSE NULL END) AS POTASSIUM
               , AVG(CASE WHEN label = 'REQUIREDO2' THEN valuenum ELSE NULL END) AS REQUIREDO2
               , AVG(CASE WHEN label = 'SODIUM' THEN valuenum ELSE NULL END) AS SODIUM
               , AVG(CASE WHEN label = 'TEMPERATURE' THEN valuenum ELSE NULL END) AS TEMPERATURE
               , AVG(CASE WHEN label = 'TIDALVOLUME' THEN valuenum ELSE NULL END) AS TIDALVOLUME
               , MAX(CASE WHEN label = 'VENTILATIONRATE' THEN valuenum ELSE NULL END) AS VENTILATIONRATE
               , MAX(CASE WHEN label = 'VENTILATOR' THEN valuenum ELSE NULL END) AS VENTILATOR
          FROM pvt
         GROUP BY pvt.subject_id, pvt.charttime
          -- remove observations if there is more than one specimen listed
          -- we do not know whether these are arterial or mixed venous, etc...
          -- happily this is a small fraction of the total number of observations
        HAVING SUM(CASE WHEN label = 'SPECIMEN' THEN 1 ELSE 0 END) < 2
        )

-- Assign an icustay_id to each of the measurment
SELECT id.icustay_id, id.hadm_id, grp.*
  FROM grp
  LEFT JOIN assign_lab_ids id
    ON grp.subject_id  = id.subject_id
   AND grp.charttime >= id.starttime
   AND grp.charttime  < id.endtime
 ORDER BY grp.subject_id, grp.charttime;

 
  DROP MATERIALIZED VIEW IF EXISTS pivoted_bg_art_mod CASCADE;
CREATE MATERIALIZED VIEW pivoted_bg_art_mod AS

  WITH stg_spo2 AS
       (
        SELECT hadm_id, charttime
            -- avg here is just used to group SpO2 by charttime
               , avg(valuenum) AS SpO2
          FROM chartevents
              -- o2 sat
        WHERE itemid IN
              (
                646 -- SpO2
               , 6719  --SpO2 -L
               , 220277 -- O2 saturation pulseoxymetry
              )
          AND valuenum > 0 AND valuenum <= 100
        GROUP BY hadm_id, charttime
        )

       , stg_fio2 AS
       (
        SELECT hadm_id, charttime
        -- pre-process the FiO2s to ensure they are between 21-100%
               , MAX(
                     CASE WHEN itemid IN (191, 3420, 3422, 223835)
                     THEN CASE
                             WHEN valuenum > 0.2 
                                   AND valuenum <= 1
                             THEN valuenum * 100
                                  -- improperly input data - looks like O2 flow in litres
                             WHEN valuenum >= 21 
                                   AND valuenum <= 100
                             THEN valuenum
                             ELSE NULL END 
                     WHEN itemid IN (189, 190, 2981, 7570) 
                           AND valuenum > 0.20 
                           AND valuenum <= 1.0
                    -- well formatted but not in %
                     THEN valuenum * 100
                     ELSE NULL END
                     ) AS fio2_chartevents
          FROM chartevents
         WHERE itemid IN
                      (
                        191
                      , 3420 -- FiO2
                      , 3422 -- FiO2 [measured]
                      , 223835 -- Inspired O2 Fraction (FiO2)
                      , 189
                      , 190 -- FiO2 set
                      , 2981
                      , 7570
                      )
           AND valuenum > 0 
           AND valuenum <= 100
                      -- exclude rows marked as error
           AND error IS DISTINCT FROM 1
         GROUP BY hadm_id, charttime
        )
        
       , stg2 AS
       (
        SELECT bg.*
               , ROW_NUMBER() OVER (PARTITION BY bg.hadm_id, bg.charttime ORDER BY s1.charttime DESC) AS lastRowSpO2
               , s1.spo2
          FROM pivoted_bg_mod bg
          LEFT JOIN stg_spo2 s1
              -- same hospitalization
            ON  bg.hadm_id = s1.hadm_id
          -- Find spo2 which occurred at most 6 hours before this blood gas 
          -- when checking next measurement of SpO2, we find that there are 3703790, 155319,
          -- 19857, 11400, 3098, 2148, 1019, 981, 466 measurements that are 1, 2, 3, 4, 5,
          -- 6, 7, 8, 9 hours after the previous reading respectively and we have decided 
          -- to truncate at 6 hours
           AND s1.charttime BETWEEN bg.charttime - INTERVAL '6' HOUR AND bg.charttime
         WHERE bg.po2 IS NOT NULL
        )
        
       , stg3 AS
       (
        SELECT bg.*
               , ROW_NUMBER() OVER (PARTITION BY bg.hadm_id, bg.charttime ORDER BY s2.charttime DESC) AS lastRowFiO2
               , s2.fio2_chartevents

               -- create our specimen prediction
               ,  1/(1+exp(-(-0.02544
               +    0.04598 * po2
               + COALESCE(-0.15356 * spo2             , -0.15356 *   97.49420 +    0.13429)
               + COALESCE( 0.00621 * fio2_chartevents ,  0.00621 *   51.49550 +   -0.24958)
               + COALESCE( 0.10559 * hemoglobin       ,  0.10559 *   10.32307 +    0.05954)
               + COALESCE( 0.13251 * so2              ,  0.13251 *   93.66539 +   -0.23172)
               + COALESCE(-0.01511 * pco2             , -0.01511 *   42.08866 +   -0.01630)
               + COALESCE( 0.01480 * fio2             ,  0.01480 *   63.97836 +   -0.31142)
               + COALESCE(-0.00200 * aado2            , -0.00200 *  442.21186 +   -0.01328)
               + COALESCE(-0.03220 * bicarbonate      , -0.03220 *   22.96894 +   -0.06535)
               + COALESCE( 0.05384 * totalco2         ,  0.05384 *   24.72632 +   -0.01405)
               + COALESCE( 0.08202 * lactate          ,  0.08202 *    3.06436 +    0.06038)
               + COALESCE( 0.10956 * ph               ,  0.10956 *    7.36233 +   -0.00617)
               + COALESCE( 0.00848 * o2flow           ,  0.00848 *    7.59362 +   -0.35803)
               ))) AS specimen_prob
          FROM stg2 bg
          LEFT JOIN stg_fio2 s2
          -- same patient
            ON bg.hadm_id = s2.hadm_id
          -- Changed to 8 hours, will impute with 0.21 unless an entry recorded here
          -- fio2 occurred at most 8 hours before this blood gas
           AND s2.charttime BETWEEN bg.charttime - INTERVAL '8' HOUR AND bg.charttime
           AND s2.fio2_chartevents > 0
         WHERE bg.lastRowSpO2 = 1 -- only the row with the most recent SpO2 (if no SpO2 found lastRowSpO2 = 1)
        )
        
SELECT stg3.hadm_id
       , stg3.icustay_id
       , stg3.charttime
       , specimen -- raw data indicating sample type, only present 80% of the time
          -- prediction of specimen for missing data
       , CASE WHEN specimen IS NOT NULL 
         THEN specimen
         WHEN specimen_prob > 0.75 THEN 'ART'
         ELSE NULL END AS specimen_pred
       , specimen_prob

        -- oxygen related parameters
       , so2, spo2 -- note spo2 is from chartevents
       , po2, pco2
       , fio2_chartevents, fio2
       , aado2
       -- also calculate AADO2
       , CASE WHEN  po2 IS NOT NULL
               AND pco2 IS NOT NULL
               AND COALESCE(fio2, fio2_chartevents) IS NOT NULL
            -- multiply by 100 because FiO2 is in a % but should be a fraction
         THEN (COALESCE(fio2, fio2_chartevents)/100) * (760 - 47) - (pco2/0.8) - po2
         ELSE NULL
          END AS aado2_calc
       , CASE WHEN po2 IS NOT NULL 
               AND COALESCE(fio2, fio2_chartevents) IS NOT NULL
           -- multiply by 100 because FiO2 is in a % but should be a fraction
         THEN 100*po2/(COALESCE(fio2, fio2_chartevents))
         --else null
         ELSE 100*po2/21
          END AS PaO2FiO2Ratio
       -- acid-base parameters
       , ph, baseexcess
       , bicarbonate, totalco2

       -- blood count parameters
       , hematocrit
       , hemoglobin
       , carboxyhemoglobin
       , methemoglobin

       -- chemistry
       , chloride, calcium
       , temperature
       , potassium, sodium
       , lactate
       , glucose

       -- ventilation stuff that's sometimes input
       , intubated, tidalvolume, ventilationrate, ventilator
       , peep, o2flow
       , requiredo2
  FROM stg3
 WHERE lastRowFiO2 = 1 -- only the most recent FiO2
  -- restrict it to *only* arterial samples
   AND (specimen = 'ART' OR specimen_prob > 0.75)
 ORDER BY hadm_id, charttime;
