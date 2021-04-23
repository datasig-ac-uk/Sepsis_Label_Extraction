/* 
This is an earlier version of what we used to extract data. Since our study set-up was agreed at the beginning of the project, our train/val/test split was based upon the data extracted with this script and therefore we have included it for completeness sake to fully reproduce the experiment.

Using the new script to produce the extraced_data table results in some 173 patient_ids not included here. None of those ids has any vital signs recorded except for glucose, and therefore it is not unreasonable for those to be excluded.
*/


  WITH cohort_info AS
       (
        SELECT ie.subject_id, ie.hadm_id, ie.icustay_id, ie.intime, ie.outtime, pat.gender
               , ROUND( (CAST(ad.admittime AS DATE) - CAST(pat.dob AS DATE))  / 365.242, 4) AS age
               , ROUND( (CAST(ie.outtime AS DATE) - CAST(ie.intime AS DATE)) , 4) AS los_icu
               , vent.starttime AS starttime_first_vent, vent.endtime AS endtime_first_vent
          FROM icustays ie
         INNER JOIN admissions ad
            ON ie.hadm_id = ad.hadm_id
         INNER JOIN patients pat
            ON ie.subject_id = pat.subject_id
          LEFT JOIN ventdurations vent
            ON vent.icustay_id = ie.icustay_id
           AND vent.ventnum = 1
         ORDER BY ie.icustay_id
        ),

       vitals AS
       (
        SELECT c.icustay_id AS icustay_id, charttime
               , MAX(CASE WHEN c.itemid IN (211, 220045, -- heart rates
                                            3450, 5815, 220047, -- alarm low, 3450 seems to be infants, 5815 adults
                                            8518, 8549, 220046 -- alarm high 
                                            -- note these alarms were removed in later versions
                                           ) 
                           AND CAST(value AS DOUBLE PRECISION) > 0 
                           AND CAST(value AS DOUBLE PRECISION) < 300 
                          THEN CAST(value AS DOUBLE PRECISION) ELSE NULL END) 
                       AS heart_rate
               , MAX(COALESCE(CASE WHEN c.itemid IN (212, 220048) THEN value ELSE NULL END)) AS heart_rhythm
               , MAX(CASE WHEN c.itemid IN (442, 455, 220179, 224167) 
                           AND CAST(value AS DOUBLE PRECISION) > 0 
                          THEN CAST(value AS DOUBLE PRECISION) ELSE NULL END) 
                       AS nbp_sys
               , MAX(CASE WHEN c.itemid IN (8440, 8441, 220180) 
                           AND CAST(value AS DOUBLE PRECISION) > 0 
                          THEN CAST(value AS DOUBLE PRECISION) ELSE NULL END) 
                       AS nbp_dias
               , MAX(CASE WHEN c.itemid IN (443, 456, 220181) 
                           AND CAST(value AS DOUBLE PRECISION) > 0 
                          THEN CAST(value AS DOUBLE PRECISION) ELSE NULL END) 
                       AS nbp_mean
               , MAX(CASE WHEN c.itemid IN (51, 6701, 220050, 225309) 
                           AND CAST(value AS DOUBLE PRECISION) > 0 
                          THEN CAST(value AS DOUBLE PRECISION) ELSE NULL END) 
                       AS abp_sys
               , MAX(CASE WHEN c.itemid IN (8364, 8368, 8555, 220051, 225310) 
                           AND CAST(value AS DOUBLE PRECISION) > 0 
                          THEN CAST(value AS DOUBLE PRECISION) ELSE NULL END) 
                       AS abp_dias
               , MAX(CASE WHEN c.itemid IN (52, 6702, 220052, 225312) 
                           AND CAST(value AS DOUBLE PRECISION) > 0 
                          THEN CAST(value AS DOUBLE PRECISION) ELSE NULL END) 
                       AS abp_mean
               , MAX(COALESCE(CASE WHEN c.itemid IN (674, 224642) 
                              THEN value ELSE NULL END)) AS temp_site
               , MAX(CASE WHEN c.itemid IN (678, 679, 3652, 3654, 6643, 227054, 223761) 
                           AND CAST(value AS DOUBLE PRECISION) > 32 
                          THEN (CAST(value AS DOUBLE PRECISION)-32.0)*5.0/9.0 -- Note 6643 only has a single data point
                          WHEN c.itemid IN (676, 677, 3655, 223762)  
                           AND CAST(value AS DOUBLE PRECISION) > 0 
                          THEN CAST(value AS DOUBLE PRECISION) ELSE NULL END) 
                       AS temp_celcius
               , MAX(CASE WHEN c.itemid IN (646, 6719, 220277, 
                                            5820, 8554, 223769, 223770, 226253 -- alarms
                                           )  
                          AND CAST(value AS DOUBLE PRECISION) > 0 
                         THEN CAST(value AS DOUBLE PRECISION) ELSE NULL END) 
                      AS o2sat
               , MAX(CASE WHEN c.itemid IN (8113, 3603, 220210,
                                            5819, 8553, 224161, 224162 -- alarms
                                           )   
                           AND CAST(value AS DOUBLE PRECISION) < 200 
                          THEN CAST(value AS DOUBLE PRECISION) 
                          WHEN c.itemid IN (618)  
                           AND valuenum < 200 
                          THEN valuenum
                          ELSE NULL END) 
                       AS resp_rate
                                           
                                           
          FROM chartevents AS c
         INNER JOIN d_items d 
            ON c.itemid = d.itemid
         WHERE c.itemid IN (
                                211, 220045, 3450, 5815, 220047, 8518, 8549, 220046,
                                212, 220048,
                                442, 455, 220179, 224167,
                                8440, 8441, 220180,
                                443, 456, 220181,
                                51, 6701, 220050, 225309,
                                8364, 8368, 8555, 220051, 225310,
                                52, 6702, 220052, 225312,
                                674, 224642,
                                678, 679, 3652, 3654, 6643, 227054, 223761,
                                676, 677, 3655, 223762,
                                646, 6719, 220277, 
                                5820, 8554, 223769, 223770, 226253,
                                618, 8113, 3603, 220210,
                                819, 8553, 224161, 224162
                            )
         GROUP BY c.icustay_id, charttime
        ),
        
       pvt AS
       ( -- begin query that extracts the data
          SELECT co.icustay_id
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
                        WHEN itemid = 50817 THEN 'SO2' -- OXYGENSATURATION
                        WHEN itemid = 50818 THEN 'PCO2'
                        WHEN itemid = 50819 THEN 'PEEP'
                        WHEN itemid = 50820 THEN 'PH'
                        WHEN itemid = 50821 THEN 'PO2'
                        WHEN itemid = 50822 THEN 'POTASSIUM'
                        WHEN itemid = 50823 THEN 'REQUIREDO2'
                        WHEN itemid = 50824 THEN 'SODIUM'
                        WHEN itemid = 50825 THEN 'TEMPERATURE'
                        WHEN itemid = 50826 THEN 'TIDALVOLUME'
                        ELSE NULL
                        END AS label
                 , charttime
                 , value
                 -- add in some sanity checks on the values
                 , CASE
                        WHEN valuenum <= 0 THEN NULL
                        WHEN itemid = 50810 AND valuenum > 100 THEN NULL -- hematocrit
                        WHEN itemid = 50816 AND valuenum > 100 THEN NULL -- FiO2
                        WHEN itemid = 50816 AND valuenum <=  0 THEN NULL -- FiO2
                        WHEN itemid = 50816 AND valuenum <=  1 THEN valuenum * 100
                        WHEN itemid = 50817 AND valuenum > 100 THEN NULL -- O2 sat
                        WHEN itemid = 50815 AND valuenum >  70 THEN NULL -- O2 flow
                        WHEN itemid = 50821 AND valuenum > 800 THEN NULL -- PO2
                        ELSE valuenum
                        END AS valuenum

            FROM cohort_info co
            LEFT JOIN labevents le
              ON le.subject_id = co.subject_id
             AND le.ITEMID IN
                              -- blood gases
                              (
                                50800, 50801, 50802, 50803, 50804, 50805, 50806, 50807, 50808, 50809
                                , 50810, 50811, 50812, 50813, 50814, 50815, 50816, 50817, 50818, 50819
                                , 50820, 50821, 50822, 50823, 50824, 50825, 50826--, 50827, 50828
                                , 51545
                              )
        ), 

       labs AS
       (
          
         SELECT pvt.ICUSTAY_ID, pvt.CHARTTIME
                , MAX(CASE WHEN label = 'SPECIMEN' THEN value ELSE NULL END) AS SPECIMEN
                , MAX(CASE WHEN label = 'AADO2' THEN valuenum ELSE NULL END) AS AADO2
                , MAX(CASE WHEN label = 'BASEEXCESS' THEN valuenum ELSE NULL END) AS BASEEXCESS
                , MAX(CASE WHEN label = 'BICARBONATE' THEN valuenum ELSE NULL END) AS BICARBONATE
                , MAX(CASE WHEN label = 'TOTALCO2' THEN valuenum ELSE NULL END) AS TOTALCO2
                , MAX(CASE WHEN label = 'CARBOXYHEMOGLOBIN' THEN valuenum ELSE NULL END) AS CARBOXYHEMOGLOBIN
                , MAX(CASE WHEN label = 'CHLORIDE' THEN valuenum ELSE NULL END) AS CHLORIDE
                , MAX(CASE WHEN label = 'CALCIUM' THEN valuenum ELSE NULL END) AS CALCIUM
                , MAX(CASE WHEN label = 'GLUCOSE' THEN valuenum ELSE NULL END) AS GLUCOSE
                , MAX(CASE WHEN label = 'HEMATOCRIT' THEN valuenum ELSE NULL END) AS HEMATOCRIT
                , MAX(CASE WHEN label = 'HEMOGLOBIN' THEN valuenum ELSE NULL END) AS HEMOGLOBIN
                , MAX(CASE WHEN label = 'INTUBATED' AND value = 'INTUBATED' THEN 1 
                           WHEN label = 'INTUBATED' AND value = 'NOT INTUBATED' THEN 1 ELSE NULL END) AS INTUBATED
                , MAX(CASE WHEN label = 'LACTATE' THEN valuenum ELSE NULL END) AS LACTATE
                , MAX(CASE WHEN label = 'METHEMOGLOBIN' THEN valuenum ELSE NULL END) AS METHEMOGLOBIN
                , MAX(CASE WHEN label = 'O2FLOW' THEN valuenum ELSE NULL END) AS O2FLOW
                , MAX(CASE WHEN label = 'FIO2' THEN valuenum ELSE NULL END) AS FIO2
                , MAX(CASE WHEN label = 'SO2' THEN valuenum ELSE NULL END) AS SO2 -- OXYGENSATURATION
                , MAX(CASE WHEN label = 'PCO2' THEN valuenum ELSE NULL END) AS PCO2
                , MAX(CASE WHEN label = 'PEEP' THEN valuenum ELSE NULL END) AS PEEP
                , MAX(CASE WHEN label = 'PH' THEN valuenum ELSE NULL END) AS PH
                , MAX(CASE WHEN label = 'PO2' THEN valuenum ELSE NULL END) AS PO2
                , MAX(CASE WHEN label = 'POTASSIUM' THEN valuenum ELSE NULL END) AS POTASSIUM
                , MAX(CASE WHEN label = 'REQUIREDO2' THEN valuenum ELSE NULL END) AS REQUIREDO2
                , MAX(CASE WHEN label = 'SODIUM' THEN valuenum ELSE NULL END) AS SODIUM
                , MAX(CASE WHEN label = 'TEMPERATURE' THEN valuenum ELSE NULL END) AS TEMPERATURE
                , MAX(CASE WHEN label = 'TIDALVOLUME' THEN valuenum ELSE NULL END) AS TIDALVOLUME

           FROM pvt
          GROUP BY pvt.icustay_id, pvt.CHARTTIME
        ),

    times AS
    (
        (SELECT icustay_id, charttime FROM vitals)
    UNION
    (SELECT icustay_id, charttime FROM labs) 
        ),
    
    merged AS
    (
        SELECT vitals2.icustay_id, vitals2.charttime, heart_rate, heart_rhythm, nbp_sys, nbp_dias, nbp_mean, abp_sys, abp_dias, abp_mean,temp_site, temp_celcius, o2sat, resp_rate, baseexcess, bicarbonate, totalco2, carboxyhemoglobin, chloride, calcium, glucose, hematocrit, hemoglobin, intubated, lactate, methemoglobin, o2flow, fio2, so2, pco2, peep, ph, po2, potassium, requiredo2, sodium, temperature, tidalvolume
        FROM
        (SELECT vitals.* 
         FROM times
        LEFT JOIN vitals
        ON times.icustay_id = vitals.icustay_id 
         AND times.charttime = vitals.charttime) vitals2
        LEFT JOIN labs
        ON vitals2.icustay_id=labs.icustay_id 
        AND vitals2.charttime = labs.charttime
    )


SELECT subject_id, hadm_id, merged.icustay_id, intime, outtime, gender, age, los_icu
       , starttime_first_vent, endtime_first_vent, merged.charttime, heart_rate, heart_rhythm
       , nbp_sys, nbp_dias, nbp_mean, abp_sys, abp_dias, abp_mean,temp_site, temp_celcius, o2sat, resp_rate
       , baseexcess, bicarbonate, totalco2, carboxyhemoglobin, chloride, calcium, glucose, hematocrit, hemoglobin
       , intubated, lactate, methemoglobin, o2flow, fio2, so2, pco2, peep, ph, po2, potassium, requiredo2, sodium
       , temperature, tidalvolume
       , (CASE WHEN merged.charttime <= endtime_first_vent 
                AND merged.charttime >= starttime_first_vent THEN 1 ELSE 0 END) AS on_vent 
  FROM cohort_info AS co
 INNER JOIN merged
    ON co.icustay_id=merged.icustay_id
 ORDER BY co.icustay_id, merged.charttime
