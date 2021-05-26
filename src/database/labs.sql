/* 
In this script we create a pivoted view of lab data that we will be using, this includes blood gases.

As labs is assigned to a hospital stay rather than icu, we look to map each reading to an icu stay.

We utilise similar ideas to the fuzzy boundaries as seen in the MIT-LCP repository (written by Alistair Johnson) but we split by hadm_id instead of subject_id to assign lags and then instead of taking halfway points, we stop associating lab data with an icustay after the outtime of the icu.  

Note that there are also labs which are outside of hopital admission time and these do not have an assigned hadm_id to the labevent, in these cases, it is possible to define a fuzzy boundary as seen in MIT-LCP, but given a lot of these are quite far away from the time the patient is in hopital, we will not do so here

We have chosen to convert some of the more common word entries into the value column into a number, we have done this only in the cases of when there are a significant number of entries of these. For code simplicity, if these are only occasionally used (e.g. <1 in carboxyhemoglobin), we do not include it here.

*/

  DROP MATERIALIZED VIEW IF EXISTS extracted_labs CASCADE;
CREATE MATERIALIZED VIEW extracted_labs AS


  -- These first 2 tables are to define a time range for each ICU stay so that any lab measurement can be assigned to one ICU stay
  WITH i AS
       (
         SELECT subject_id, hadm_id, icustay_id, intime, outtime
                -- find the outtime of the previous icustay and the intime of the next icustay
                , LAG (outtime) OVER (PARTITION BY subject_id, hadm_id ORDER BY intime) AS outtime_lag
                , LEAD (intime) OVER (PARTITION BY subject_id, hadm_id ORDER BY intime) AS intime_lead
           FROM icustays
        )

       , iid_assign AS
       (
         SELECT i.subject_id, i.hadm_id, i.icustay_id
                , admittime, dischtime,  intime, outtime, outtime_lag, intime_lead -- this line is just for checking purposes, can comment out
                -- for the first icustay, everything starting from hospital admission to outtime is assigned to this icustay id
                -- for subsequent icustays, only data from the end of the previous icustay to the end of the currect icustay gets assigned to these ids. For the last icustay in the sequence, then everything from icu intime to the end of hospital admission is included
                -- in order for the data extraction to be as inclusive as possible almost all lab events are included using the following boundaries (3 days before and after discharge - actually only 2.5 days / 60 hours before is required but this is easier for the purposed of having a hospital admission boundary). Only leaves out a special case which has been raised as an issue. The discrepency between edregtime and admittime has been raised as an issue.
                , CASE WHEN i.outtime_lag IS NOT NULL THEN i.outtime_lag
                  ELSE ad.admittime - INTERVAL '72' HOUR
                  END AS data_start
                , CASE WHEN i.intime_lead IS NOT NULL THEN i.outtime
                  ELSE ad.dischtime + INTERVAL '72' HOUR
                  END AS data_end
           FROM i
          INNER JOIN admissions ad 
             ON i.hadm_id = ad.hadm_id
        )
        
        -- Next  also create fuzzy boundaries on hospitalization
        -- Note that sometimes the discharge time of the previous hospital occurs after the next hospital admission causing a negative difference, in that case the boundary here is set before the discharge time of the previous stay/after admittime of next hospital stay
        -- No duplicate data appears to be created if we drop this requirement and this is likely to how the labevents table was created in the first place.
       , h AS
            (
             SELECT subject_id, hadm_id, admittime, dischtime
                    , LAG (dischtime) OVER (PARTITION BY subject_id ORDER BY admittime) AS dischtime_lag
                    , LEAD (admittime) OVER (PARTITION BY subject_id ORDER BY admittime) AS admittime_lead
               FROM admissions
        )
        
       , adm AS
        (
         SELECT h.subject_id, h.hadm_id
            -- this rule is:
            --  if there are two hospitalizations within 24 hours, set the start/stop
            --  time as half way between the two admissions
                , CASE WHEN h.dischtime_lag IS NOT NULL
                            AND h.dischtime_lag > (h.admittime - INTERVAL '144' HOUR)
                       THEN h.admittime - ((h.admittime - h.dischtime_lag)/2)
                       ELSE h.admittime - INTERVAL '72' HOUR
                       END AS data_start
                , CASE WHEN h.admittime_lead IS NOT NULL
                            AND h.admittime_lead < (h.dischtime + INTERVAL '144' HOUR)
                       THEN h.dischtime + ((h.admittime_lead - h.dischtime)/2)
                       ELSE (h.dischtime + INTERVAL '72' HOUR)
                       END AS data_end
          FROM h
        )

        -- Now we pivot the data, to get a chart time then a column for each of the lab variables
        -- Below we also include some information on units, and common non-numerical values
       , grp AS
        (
         SELECT subject_id, charttime
                , MAX(CASE WHEN itemid = 50800 
                      THEN value 
                      ELSE NULL END) AS specimen
                , MAX(CASE WHEN itemid = 50801 
                            AND valuenum > 0 
                            AND valuenum < 800  
                      THEN valuenum ELSE NULL END) AS aado2-- chosen to include all except 1 negative value
                , MAX(CASE WHEN itemid = 50802 
                            AND valuenum > -40 
                            AND valuenum < 40  
                      THEN valuenum 
                      ELSE NULL END) AS baseexcess
                , MAX(CASE WHEN itemid = 50803 
                            AND valuenum >= 0 
                            AND valuenum < 60 
                      THEN valuenum 
                      ELSE NULL END) AS bicarbonate_bg
                , MAX(CASE WHEN itemid = 50804 
                            AND valuenum >= 0 
                            AND valuenum < 120 
                      THEN valuenum 
                      ELSE NULL END) AS totalco2 -- = 0.23*pCO2 + bicarbonate
                , MAX(CASE WHEN itemid = 50805 
                            AND valuenum >= 0 
                            AND valuenum < 20  
                      THEN valuenum 
                      ELSE NULL END) AS carboxyhemoglobin -- '<1.0' 'LESS THAN 1' '<1'
                , MAX(CASE WHEN itemid = 50806 
                            AND valuenum > 50 
                            AND valuenum < 160 
                      THEN valuenum 
                      ELSE NULL END) AS chloride_bg -- mEq/L
                , MAX(CASE WHEN itemid = 50808 
                            AND valuenum > 0 
                            AND valuenum < 4 
                      THEN valuenum/0.2495 
                      ELSE NULL END) AS calcium_bg -- Measured in mmol/L
                , MAX(CASE WHEN itemid = 50809 
                            AND valuenum > 0 
                            AND valuenum < 1300 
                      THEN valuenum 
                      ELSE NULL END) AS glucose_bg -- GREATER THAN 500 has 91 values. mg/dL to convert to mmol/L divide by 18.0182
                , MAX(CASE WHEN itemid = 50810 
                            AND valuenum >= 0 
                            AND valuenum < 80 
                      THEN valuenum 
                      ELSE NULL END) AS hematocrit_bg --%
                , MAX(CASE WHEN itemid = 50811 
                            AND valuenum > 0 
                            AND valuenum < 30 
                      THEN valuenum 
                      ELSE NULL END) AS hemoglobin_bg -- g/dL
                , MAX(CASE WHEN itemid = 50812 
                            AND value = 'INTUBATED' 
                      THEN 1 
                      WHEN itemid = 50812 
                            AND value = 'NOT INTUBATED' 
                      THEN 0 ELSE NULL END) AS intubated
                , MAX(CASE WHEN itemid = 50813 
                            AND valuenum >= 0 
                            AND valuenum < 40 
                      THEN valuenum 
                      ELSE NULL END) AS lactate
                , MAX(CASE WHEN itemid = 50814 
                            AND valuenum >= 0 
                            AND valuenum < 100 
                      THEN valuenum 
                      ELSE NULL END) AS methemoglobin --- 'LESS THAN 1' '<1.0' '<1'
                , MAX(CASE WHEN itemid = 50815 
                            AND valuenum >= 0 
                            AND valuenum <= 100 
                      THEN valuenum 
                      ELSE NULL END) AS o2flow
                , MAX(CASE WHEN itemid = 50816 
                            AND valuenum >= 20 
                            AND valuenum <= 100 
                      THEN valuenum 
                      ELSE NULL END) AS fio2
                , MAX(CASE WHEN itemid = 50817 
                            AND valuenum >= 0 
                            AND valuenum <= 100 
                      THEN valuenum 
                      ELSE NULL END) AS so2
                , MAX(CASE WHEN itemid = 50818 
                            AND valuenum >= 0 
                            AND valuenum < 250 
                      THEN valuenum 
                      ELSE NULL END) AS pco2 --1 kPa = 7.50062 mmHg
                , MAX(CASE WHEN itemid = 50819 
                            AND valuenum >= 0 
                            AND valuenum <= 50 
                      THEN valuenum 
                      ELSE NULL END) AS peep
                , MAX(CASE WHEN itemid = 50820 
                            AND valuenum > 6.2 
                            AND valuenum < 8 
                      THEN valuenum 
                      ELSE NULL END) AS ph
                , MAX(CASE WHEN itemid = 50821 
                            AND valuenum >= 0 
                            AND valuenum <= 800 
                      THEN valuenum 
                      ELSE NULL END) AS po2
                , MAX(CASE WHEN itemid = 50822 
                            AND valuenum >= 1.4 
                            AND valuenum <= 12  
                      THEN valuenum 
                      ELSE NULL END) AS potassium_bg --'GREATER THAN 10.0' '>10.0' '>10' 'GREATER 10.0'
                , MAX(CASE WHEN itemid = 50823 
                            AND valuenum >= 20 
                            AND valuenum <= 100  
                      THEN valuenum 
                      ELSE NULL END) AS requiredo2
                , MAX(CASE WHEN itemid = 50824 
                            AND valuenum > 80 
                            AND valuenum < 185 
                      THEN valuenum 
                      ELSE NULL END) AS sodium_bg
                , MAX(CASE WHEN itemid = 50825 
                            AND valuenum > 15 
                            AND valuenum < 45 
                      THEN valuenum 
                      ELSE NULL END) AS temperature
                , MAX(CASE WHEN itemid = 50826  
                            AND valuenum >= 0 
                            AND valuenum < 5000 
                      THEN valuenum 
                      ELSE NULL END) AS tidalvolume 
                , MAX(CASE WHEN itemid = 50828 
                      THEN value 
                      ELSE NULL END) AS ventilator
                , MAX(CASE WHEN itemid = 50863 
                            AND valuenum >= 0 
                            AND valuenum < 5000 
                      THEN valuenum 
                      ELSE NULL END) AS alkalinephos -- Note this actually does not filter out any values
                , MAX(CASE WHEN itemid = 50878 
                      THEN valuenum 
                      ELSE NULL END) AS ast -- This does not seem to have any upper bound observed in the data up to 36400 (repeated measurement)
                , MAX(CASE WHEN itemid = 50883 
                            AND valuenum >= 0 
                            AND valuenum <= 72 
                      THEN valuenum 
                      ELSE NULL END) AS bilirubin_direct -- To convert results from mg/dL to μmol/L, multiply mg/dL by 17.1. 29.2
                        -- 72 are the highest --'<0.1' 20 obs 
                , MAX(CASE WHEN itemid = 50885 
                            AND valuenum >= 0 
                            AND valuenum < 83 
                      THEN valuenum 
                      ELSE NULL END) AS bilirubin_total -- 82.8 highest 'LESS THAN 2.0' 11
                , MAX(CASE WHEN itemid = 51006 
                            AND valuenum > 0 
                            AND valuenum < 300 
                      THEN valuenum 
                      ELSE NULL END) AS bun --  mg/dL to mmol/L *0.3571 
                , MAX(CASE WHEN itemid = 50912 
                            AND valuenum > 0 
                            AND valuenum < 45 
                      THEN valuenum 
                      ELSE NULL END) AS creatinine --One mg/dL of creatinine is 88.4 μmol/L '<0.1' 18, 'LESS THAN 0.5' 18, 'LESS THAN 0.4', 15, 'LESS THAN 0.3', 14, 'LESS THAN 0.2', 9, 'LESS THAN 1.0', 6, 'LESS THAN 0.1', 4, 'LESS THAN 0.6', 4
                , MAX(CASE WHEN itemid = 51214 
                            AND valuenum > 0 
                            AND valuenum < 1800 
                      THEN valuenum 
                      ELSE NULL END) AS fibrinogen -- 'LESS THAN 35' 11
    -- again this seems to be a variable that just increments. Only 7 patients between 1400 and 1800
                , MAX(CASE WHEN itemid = 50960 
                            AND valuenum > 0 
                            AND valuenum < 20 
                      THEN valuenum 
                      ELSE NULL END) AS magnesium -- set to 13 if want to include last repeat
                , MAX(CASE WHEN itemid = 50970 
                            AND valuenum > 0 
                            AND valuenum < 20 
                      THEN valuenum 
                      ELSE NULL END) AS phosphate -- There are a few values between 20 and 30, 4 in 30s, then 1 value in 50s
    -- '<0.3' 6, 'LESS THAN 1.0', 3
                , MAX(CASE WHEN itemid = 51265 
                            AND valuenum >= 0 
                            AND valuenum < 2000 
                      THEN valuenum 
                      ELSE NULL END) AS platelets -- '<5', 78
                , MAX(CASE WHEN itemid = 51275 
                            AND value IN ('>150', '150 IS HIGHEST MEASURED PTT') 
                      THEN 160
                      WHEN itemid = 51275 
                            AND value NOT IN ('>150', '150 IS HIGHEST MEASURED PTT') 
                            AND valuenum > 0 
                            AND valuenum <= 150  
                      THEN valuenum 
                      ELSE NULL END) AS ptt --filters out 1 with value 0  -- '>150' 1206 '150 IS HIGHEST MEASURED PTT', 169
                , MAX(CASE WHEN itemid = 51002 
                            AND value IN ('GREATER THAN 50', 'GREATER THAN 50.0', '>50', 'GREATER THAN 50 NG/ML', 'GREATER THAN FIFTY') 
                      THEN 51
                      WHEN itemid = 51002 
                            AND value IN ('<0.3', 'LESS THAN 0.3') 
                      THEN 0.2
                      WHEN itemid = 51002 
                            AND value NOT IN ('GREATER THAN 50', 'GREATER THAN 50.0', '>50', 'GREATER THAN 50 NG/ML', 'GREATER THAN FIFTY', '<0.3', 'LESS THAN 0.3') 
                            AND valuenum >= 0 
                            AND valuenum <= 50 THEN valuenum ELSE NULL END) AS tropinin_t
                , MAX(CASE WHEN itemid = 51003 
                            AND value IN ('<0.01', 'LESS THAN 0.01') THEN 0.005
                      WHEN itemid = 51003 
                            AND value NOT IN ('<0.01', 'LESS THAN 0.01') 
                            AND valuenum >= 0 
                            AND valuenum <= 30 
                      THEN valuenum 
                      ELSE NULL END) AS tropinin_i -- technically the range does nothing in this case
                , MAX(CASE WHEN itemid IN (51300, 51301) 
                            AND value = '<0.1' 
                      THEN 0.05
                      WHEN itemid IN (51300, 51301) 
                            AND value != '<0.1' 
                            AND valuenum <  700 
                      THEN valuenum ELSE NULL END) AS wbc --'<0.1'   314
                , MAX(CASE WHEN itemid = 50882 
                            AND valuenum >= 0 
                            AND valuenum < 60 
                      THEN valuenum 
                      ELSE NULL END) AS bicarbonate -- there are 45 cases of 'LESS THAN 5' and 44 cases of 'GREATER THAN 50', '<5', 22, 'LESS THAN 5.0' 5 '<5.0' 4, '>50' 4, 'GREATER THAN 45', 4 'GREATER 50' 1 
                , MAX(CASE WHEN itemid = 50902 
                            AND valuenum > 50 
                            AND valuenum < 160 
                      THEN valuenum 
                      ELSE NULL END) AS chloride -- there are a few word entries here 'GREATER THAN 140', 11, '>140' 'LESS THAN 60' 2
                , MAX(CASE WHEN itemid = 50893 
                            AND valuenum > 0 
                            AND valuenum < 30 
                      THEN valuenum 
                      ELSE NULL END) AS calcium -- measured in mg/dL therefore to convert on same scale as blood gas version *0.2495
                , MAX(CASE WHEN itemid = 50931 
                            AND valuenum > 0 
                            AND valuenum < 1300 
                      THEN valuenum 
                      ELSE NULL END) AS glucose -- up to 3200 for repeat
                , MAX(CASE WHEN itemid = 51221 
                            AND valuenum >= 0 
                            AND valuenum <= 100 
                      THEN valuenum 
                      ELSE NULL END) AS hematocrit
                , MAX(CASE WHEN itemid = 51222 
                            AND valuenum > 0 
                            AND valuenum < 30 
                      THEN valuenum 
                      ELSE NULL END) AS hemoglobin
                , MAX(CASE WHEN itemid = 50971 
                            AND valuenum >= 1.4 
                            AND valuenum <= 12 
                      THEN valuenum 
                      ELSE NULL END) AS potassium -- 'GREATER THAN 10' 190 'GREATER THAN 10.0' 82 '>10.0' 10 '>10' 9 '> 10' 2 'GREATER THEN 10' 2 '>GREATER THAN 10' 1
                , MAX(CASE WHEN itemid = 50983 
                            AND valuenum > 80 
                            AND valuenum < 185 
                      THEN valuenum 
                      ELSE NULL END) AS sodium -- 'GREATER THAN 180' 7 '>180' 1
           FROM labevents le
          WHERE le.ITEMID IN
                      -- all the ids for blood gases
                      (
                        50800, 50801, 50802, 50803, 50804, 50805, 50806, 50807, 50808, 50809
                        , 50810, 50811, 50812, 50813, 50814, 50815, 50816, 50817, 50818, 50819
                        , 50820, 50821, 50822, 50823, 50824, 50825, 50826, 50828, 50863, 50878
                        , 50883, 50885, 51006, 50912, 51214, 50960, 50970, 51265, 51275, 51002
                        , 51003, 51300, 51301, 50882, 50902, 50893, 50931, 51221, 51222, 50971, 50983
                      )
          GROUP BY subject_id, charttime
            -- only take those cases where there is just one specimen, otherwise it is not clear whether we are getting arterial readings or venous.
         HAVING SUM(CASE WHEN itemid = 50800 
                    THEN 1 
                    ELSE 0 END) < 2
        )

        -- Join the above data so that they have an icustay_id
        , labs AS
        (
         SELECT iid.icustay_id, grp.* 
           FROM grp
           LEFT JOIN adm
             ON grp.subject_id = adm.subject_id
            AND grp.charttime >= adm.data_start
            AND grp.charttime < adm.data_end   
          INNER JOIN iid_assign iid
             ON grp.subject_id = iid.subject_id
            AND iid.hadm_id = adm.hadm_id
          WHERE grp.charttime >= iid.data_start 
            AND grp.charttime <= iid.data_end
            AND (grp.specimen = 'ART' 
             OR grp.specimen IS NULL)
          ORDER BY grp.subject_id, grp.charttime
        )
        
SELECT * FROM labs

